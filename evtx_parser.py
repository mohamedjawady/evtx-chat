import os
import sys
import json
import csv
import xml.dom.minidom
import subprocess
from typing import Dict, List, Any, Optional, Union

try:
    import evtx
    HAS_EVTX = True
except ImportError:
    HAS_EVTX = False

try:
    import win32evtlogutil
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

class EvtxParser:
    """Class for parsing and extracting data from EVTX files"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EVTX file not found: {file_path}")
        self.message_cache = {}

    def extract_text(self) -> str:
        """Extract text from EVTX file for RAG processing"""
        try:
            records = self.get_records(debug=False, render_messages=True)
            text_blocks = []

            for record in records:
                # Add basic event info
                event_info = [
                    f"Event ID: {record['System']['EventID']['Value']}",
                    f"Time: {record['System']['TimeCreated']['SystemTime']}",
                    f"Channel: {record['System']['Channel']}",
                    f"Computer: {record['System']['Computer']}"
                ]

                # Add rendered message if available
                if 'RenderedMessage' in record:
                    event_info.append(f"Message: {record['RenderedMessage']}")

                # Add EventData if available
                if 'EventData' in record:
                    for key, value in record['EventData'].items():
                        if value:  # Only add non-empty values
                            event_info.append(f"{key}: {value}")

                text_blocks.append('\n'.join(event_info))

            return '\n\n'.join(text_blocks)
        except Exception as e:
            return f"Error extracting text from EVTX file: {str(e)}"

    def get_records(self, debug: bool = False, render_messages: bool = False) -> List[Dict[str, Any]]:
        """Retrieves event log records from the EVTX file."""
        if HAS_EVTX:
            with evtx.Evtx(self.file_path) as log:
                records = []
                for record in log.records():
                    try:
                        event_record = self.parse_evtx_record(record, render_messages)
                        if event_record:
                            records.append(event_record)
                    except Exception as e:
                        if debug:
                            print(f"Error parsing record: {e}", file=sys.stderr)
                return records
        elif HAS_WIN32:
            return self._get_records_win32(render_messages)
        else:
            return []


    def _get_records_win32(self, render_messages: bool = False) -> List[Dict[str, Any]]:
        """Retrieves records using the win32evtlogutil library."""
        try:
            records = []
            for record in win32evtlogutil.EvtLogRecordIterator(self.file_path):
                try:
                    event_record = self._parse_win32_record(record, render_messages)
                    if event_record:
                        records.append(event_record)
                except Exception as e:
                    print(f"Error parsing record with win32evtlogutil: {e}", file=sys.stderr)
            return records

        except Exception as e:
            print(f"Error accessing log file with win32evtlogutil: {e}", file=sys.stderr)
            return []

    def _parse_win32_record(self, record, render_messages: bool = False) -> Optional[Dict[str, Any]]:
        """Parse a single record obtained using win32evtlogutil."""
        event_record = {}
        event_record["System"] = {
            "EventID": {"Value": record.EventID},
            "TimeCreated": {"SystemTime": record.TimeCreated.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"},
            "Channel": record.StringInserts[0], # Assuming Channel is the first insert
            "Computer": record.ComputerName
        }

        if render_messages and record.RenderedDescription:
            event_record["RenderedMessage"] = record.RenderedDescription
        if record.Data:
            event_record["EventData"] = {str(k):v for k,v in enumerate(record.Data)}

        return event_record


    def parse_evtx_record(self, record, render_messages: bool = False) -> Optional[Dict[str, Any]]:
        """Parse a single record from an EVTX file using python-evtx."""
        try:
            event_record = {}
            system_fields = {
                "EventID": "Value",
                "TimeCreated": "SystemTime",
                "Channel": "Channel",
                "Computer": "Computer"
            }

            event_record["System"] = {}
            for field, attr in system_fields.items():
                if getattr(record, field):
                    event_record["System"][field] = {attr:getattr(record, field)}

            if render_messages and record.strings and len(record.strings) > 0:
                event_record["RenderedMessage"] = record.strings[0]

            if record.data:
                event_record["EventData"] = self.parse_event_data(record.data)

            return event_record
        except Exception as e:
            print(f"Error parsing record: {e}", file=sys.stderr)
            return None

    def parse_event_data(self, event_data) -> Dict[str, Any]:
        """Parse the EventData section of an EVTX record."""
        parsed_data = {}
        for item in event_data:
            name = item.name
            value = item.value
            if isinstance(value, bytes):
                value = value.decode('utf-8', 'ignore')
            parsed_data[name] = value
        return parsed_data

    def get_message(self, record, event_id: int) -> str:
        """Retrieves the message string for a given event ID from the message cache."""
        if event_id not in self.message_cache:
            try:
                message = self.get_message_from_file(record, event_id)
                self.message_cache[event_id] = message
            except Exception as e:
                print(f"Error getting message: {e}", file=sys.stderr)
                return "Message not found"
        return self.message_cache[event_id]

    def get_message_from_file(self, record, event_id: int) -> str:
        """Retrieves message from the EVTX file using subprocess call."""

        # Construct the command to extract message
        cmd = [
            "powershell.exe",
            "-Command",
            f"""
            $log = Get-WinEvent -ListLog * -MaxEvents 1
            Get-WinEvent -LogName $log.LogDisplayName -EventID {event_id} | Select-Object -ExpandProperty Message
            """
        ]

        try:
            # Execute the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            message = result.stdout.strip()
            return message
        except subprocess.CalledProcessError as e:
            print(f"Error executing PowerShell command: {e}", file=sys.stderr)
            return "Message not found"
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return "Message not found"



def evtx_to_text(file_path: str) -> str:
    """Convert an EVTX file to text format for RAG processing"""
    try:
        parser = EvtxParser(file_path)
        return parser.extract_text()
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:
        return f"Error converting EVTX to text: {str(e)}"