
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

    def get_records(self, limit: Optional[int] = None, debug: bool = False, 
                   render_messages: bool = True) -> List[Dict[str, Any]]:
        """
        Extract records from the EVTX file

        Args:
            limit: Maximum number of records to extract (None for all)
            debug: If True, print the XML for each record for debugging
            render_messages: If True, attempt to retrieve rendered message text

        Returns:
            List of dictionaries containing event data
        """
        records = []
        count = 0

        try:
            with evtx.Evtx(self.file_path) as evtx_file:
                for record in evtx_file.records():
                    if limit is not None and count >= limit:
                        break

                    try:
                        xml_str = record.xml()

                        if debug:
                            print(f"\n--- Record {record.record_num()} XML ---")
                            print(xml_str)
                            print("-----------------------------\n")

                        event_data = self._parse_xml_to_dict(xml_str)
                        event_data['record_num'] = record.record_num()

                        if render_messages and HAS_WIN32:
                            self._add_rendered_message(event_data, debug)

                        records.append(event_data)
                        count += 1
                    except Exception as e:
                        print(f"Error parsing record {record.record_num()}: {e}", file=sys.stderr)
                        if debug:
                            import traceback
                            traceback.print_exc()
                        continue
        except Exception as e:
            print(f"Error reading EVTX file: {e}", file=sys.stderr)
            if debug:
                import traceback
                traceback.print_exc()

        return records

    def _parse_xml_to_dict(self, xml_str: str) -> Dict[str, Any]:
        """
        Parse XML string to dictionary

        Args:
            xml_str: XML string from EVTX record

        Returns:
            Dictionary with extracted event data
        """
        dom = xml.dom.minidom.parseString(xml_str)
        event = dom.getElementsByTagName("Event")[0]

        system = event.getElementsByTagName("System")[0]

        event_id_element = system.getElementsByTagName("EventID")
        event_id = ""
        qualifiers = ""
        if event_id_element:
            event_id = self._get_element_text(system, "EventID")
            qualifiers = event_id_element[0].getAttribute("Qualifiers")

        provider_element = system.getElementsByTagName("Provider")
        provider_name = ""
        provider_guid = ""
        event_source_name = ""
        if provider_element:
            provider_name = provider_element[0].getAttribute("Name")
            provider_guid = provider_element[0].getAttribute("Guid")
            event_source_name = provider_element[0].getAttribute("EventSourceName")

        result = {
            "System": {
                "Provider": {
                    "Name": provider_name,
                    "Guid": provider_guid,
                    "EventSourceName": event_source_name
                },
                "EventID": {
                    "Value": event_id,
                    "Qualifiers": qualifiers
                },
                "Version": self._get_element_text(system, "Version"),
                "Level": self._get_element_text(system, "Level"),
                "Task": self._get_element_text(system, "Task"),
                "Opcode": self._get_element_text(system, "Opcode"),
                "Keywords": self._get_element_text(system, "Keywords"),
                "TimeCreated": {
                    "SystemTime": self._get_element_attribute(system, "TimeCreated", "SystemTime")
                },
                "EventRecordID": self._get_element_text(system, "EventRecordID"),
                "Channel": self._get_element_text(system, "Channel"),
                "Computer": self._get_element_text(system, "Computer"),
            }
        }

        event_data_nodes = event.getElementsByTagName("EventData")
        if event_data_nodes:
            event_data_element = event_data_nodes[0]
            data_elements = event_data_element.getElementsByTagName("Data")

            if data_elements:
                event_data = {}
                for idx, data in enumerate(data_elements):
                    name = data.getAttribute("Name")
                    value = data.firstChild.nodeValue if data.firstChild else ""

                    if name:
                        event_data[name] = value
                    else:
                        event_data[f"Data_{idx}"] = value

                result["EventData"] = event_data
            else:
                text_content = event_data_element.textContent.strip()
                if text_content:
                    lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                    result["EventData"] = {f"Data_{i}": value for i, value in enumerate(lines)}

        return result

    def _get_element_text(self, parent, tag_name: str) -> str:
        """Extract text from an XML element"""
        elements = parent.getElementsByTagName(tag_name)
        if elements and elements[0].firstChild:
            return elements[0].firstChild.nodeValue
        return ""

    def _get_element_attribute(self, parent, tag_name: str, attr_name: str) -> str:
        """Extract attribute value from an XML element"""
        elements = parent.getElementsByTagName(tag_name)
        if elements:
            return elements[0].getAttribute(attr_name)
        return ""
