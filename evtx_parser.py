"""
Enhanced EVTX Parser for Threat Hunting RAG

This module provides functionality to parse Windows Event Log (EVTX) files 
and convert them to text format for use in RAG systems.
"""

import os
import re
import xml.dom.minidom
import logging
import json
from typing import Dict, List, Any, Optional, Set, Union

class SimpleEvtxParser:
    """A simplified parser for EVTX files that doesn't rely on external dependencies"""
    
    def __init__(self, file_path: str):
        """
        Initialize the EVTX parser

        Args:
            file_path: Path to the EVTX file
        """
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EVTX file not found: {file_path}")
    
    def extract_text(self) -> str:
        """
        Extract text content from an EVTX file for use in RAG systems.
        
        This is a simplified method that tries to extract basic XML content
        without requiring python-evtx library.
        
        Returns:
            A string containing the extracted text
        """
        try:
            # Try with xml.dom.minidom if it's a valid XML file
            try:
                with open(self.file_path, 'rb') as f:
                    content = f.read(8192)  # Read the first 8KB to check if it's binary
                
                # Check if file appears to be a binary EVTX file (has ElfFile signature)
                if b'ElfFile' in content or content.startswith(b'\x45\x6c\x66\x46'):
                    return f"The EVTX file {os.path.basename(self.file_path)} is in binary format and requires special parsing.\n\nBinary EVTX files contain Windows Event Log records which may include security events, system events, application logs, and other Windows event data."
                
                # If not binary, try parsing as XML
                with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    xml_content = f.read()
                
                dom = xml.dom.minidom.parseString(xml_content)
                events = []
                
                # Handle regular Event logs (Windows format)
                for event in dom.getElementsByTagName("Event"):
                    event_text = self._extract_event_text(event)
                    if event_text:
                        events.append(event_text)
                
                # If no events found, try other XML formats
                if not events:
                    # Try looking for custom formats like Events wrapper
                    events_nodes = dom.getElementsByTagName("Events")
                    if events_nodes:
                        for event in events_nodes[0].getElementsByTagName("Event"):
                            event_text = self._extract_event_text(event)
                            if event_text:
                                events.append(event_text)
                
                if events:
                    return "\n\n".join(events)
                else:
                    # If we found XML but no events, return a general summary
                    root_elements = [node.nodeName for node in dom.documentElement.childNodes 
                                   if node.nodeType == node.ELEMENT_NODE]
                    return f"XML file found with root elements: {', '.join(root_elements)}\n\nFile appears to be in XML format but doesn't contain standard Windows Event Log structures."
                
            except Exception as xml_error:
                logging.warning(f"XML parsing failed: {xml_error}")
                
                # Attempt text extraction for non-XML files
                with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                # Extract text-like content
                text_blocks = []
                current_block = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_block:
                            text_blocks.append(" ".join(current_block))
                            current_block = []
                    # Only include likely text content
                    elif re.match(r'^[a-zA-Z0-9\s.,;:\'"\-_()[\]{}@#$%^&*+=|\\/<>!?]+$', line):
                        current_block.append(line)
                
                if current_block:
                    text_blocks.append(" ".join(current_block))
                
                if text_blocks:
                    return "\n\n".join(text_blocks)
                else:
                    return f"Could not extract readable text from {os.path.basename(self.file_path)}. The file appears to be in a binary or non-text format."
        
        except Exception as e:
            logging.error(f"Error extracting text from EVTX file: {e}")
            return f"Error extracting text from file: {str(e)}"
    
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
    
    def _parse_user_data_node(self, user_data_node) -> Dict[str, Any]:
        """Parse UserData node which can contain custom XML structures"""
        result = {}
        
        for child in user_data_node.childNodes:
            if child.nodeType == child.ELEMENT_NODE:
                element_data = {}
                
                # Get attributes
                for attr_name, attr_value in child.attributes.items():
                    element_data[f"@{attr_name}"] = attr_value
                
                # Get child elements recursively
                for sub_child in child.childNodes:
                    if sub_child.nodeType == sub_child.ELEMENT_NODE:
                        has_children = False
                        for n in sub_child.childNodes:
                            if n.nodeType == n.ELEMENT_NODE:
                                has_children = True
                                break
                        
                        if has_children:
                            element_data[sub_child.nodeName] = self._parse_user_data_node(sub_child)
                        elif sub_child.firstChild:
                            element_data[sub_child.nodeName] = sub_child.firstChild.nodeValue
                
                # Store the element data
                if not element_data and child.firstChild:
                    result[child.nodeName] = child.firstChild.nodeValue
                else:
                    result[child.nodeName] = element_data
        
        return result
    
    def _extract_event_text(self, event_node) -> str:
        """
        Extract human-readable text from an Event XML node
        
        Args:
            event_node: XML node containing an Event
            
        Returns:
            String with readable event information
        """
        try:
            # Initialize the event text parts
            event_parts = []
            
            # Extract System metadata
            system_nodes = event_node.getElementsByTagName("System")
            if system_nodes:
                system = system_nodes[0]
                
                # Get Event Record ID for reference
                event_record_id = self._get_element_text(system, "EventRecordID")
                if event_record_id:
                    event_parts.append(f"Event Record ID: {event_record_id}")
                
                # Get provider info
                provider_elements = system.getElementsByTagName("Provider")
                if provider_elements:
                    provider = provider_elements[0]
                    provider_name = provider.getAttribute("Name")
                    if provider_name:
                        event_parts.append(f"Provider: {provider_name}")
                    
                    # Get provider GUID if available
                    provider_guid = provider.getAttribute("Guid")
                    if provider_guid:
                        event_parts.append(f"Provider GUID: {provider_guid}")
                
                # Get Event ID
                event_id_elements = system.getElementsByTagName("EventID")
                if event_id_elements:
                    event_id = self._get_element_text(system, "EventID")
                    if event_id:
                        event_parts.append(f"Event ID: {event_id}")
                    
                    # Get qualifiers if available
                    qualifiers = event_id_elements[0].getAttribute("Qualifiers")
                    if qualifiers:
                        event_parts.append(f"Qualifiers: {qualifiers}")
                
                # Get Level
                level = self._get_element_text(system, "Level")
                if level:
                    level_display = level
                    # Translate common level numbers to names
                    if level == "0": level_display = "0 (LogAlways)"
                    elif level == "1": level_display = "1 (Critical)"
                    elif level == "2": level_display = "2 (Error)"
                    elif level == "3": level_display = "3 (Warning)"
                    elif level == "4": level_display = "4 (Informational)"
                    elif level == "5": level_display = "5 (Verbose)"
                    event_parts.append(f"Level: {level_display}")
                
                # Get Task and Keywords
                task = self._get_element_text(system, "Task")
                if task:
                    event_parts.append(f"Task: {task}")
                
                keywords = self._get_element_text(system, "Keywords")
                if keywords:
                    event_parts.append(f"Keywords: {keywords}")
                
                # Get Channel
                channel = self._get_element_text(system, "Channel")
                if channel:
                    event_parts.append(f"Channel: {channel}")
                
                # Get time created
                time_elements = system.getElementsByTagName("TimeCreated")
                if time_elements:
                    time_created = time_elements[0].getAttribute("SystemTime")
                    if time_created:
                        event_parts.append(f"Time: {time_created}")
                
                # Get computer name
                computer = self._get_element_text(system, "Computer")
                if computer:
                    event_parts.append(f"Computer: {computer}")
                
                # Get Security UserID if present
                security_elements = system.getElementsByTagName("Security")
                if security_elements:
                    user_id = security_elements[0].getAttribute("UserID")
                    if user_id:
                        event_parts.append(f"User ID: {user_id}")
            
            # Extract Event Data (structured data)
            event_data_nodes = event_node.getElementsByTagName("EventData")
            if event_data_nodes:
                event_data_elements = event_data_nodes[0].getElementsByTagName("Data")
                
                if event_data_elements:
                    event_parts.append("\nEvent Data:")
                    
                    for data_element in event_data_elements:
                        name = data_element.getAttribute("Name")
                        value = data_element.firstChild.nodeValue if data_element.firstChild else ""
                        
                        if name and value:
                            event_parts.append(f"  {name}: {value}")
                        elif value:
                            event_parts.append(f"  {value}")
                else:
                    # Check for raw textContent if no Data elements
                    text_content = event_data_nodes[0].textContent.strip()
                    if text_content:
                        event_parts.append("\nEvent Data (Raw):")
                        event_parts.append(f"  {text_content}")
            
            # Extract UserData (for application-specific data)
            user_data_nodes = event_node.getElementsByTagName("UserData")
            if user_data_nodes:
                event_parts.append("\nUser Data:")
                user_data_dict = self._parse_user_data_node(user_data_nodes[0])
                
                # Format the user data dict as a readable string
                for key, value in user_data_dict.items():
                    if isinstance(value, dict):
                        event_parts.append(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            event_parts.append(f"    {sub_key}: {sub_value}")
                    else:
                        event_parts.append(f"  {key}: {value}")
            
            # Extract RenderingInfo if available
            rendering_info_nodes = event_node.getElementsByTagName("RenderingInfo")
            if rendering_info_nodes:
                event_parts.append("\nRendering Info:")
                
                # Get Message
                message_nodes = rendering_info_nodes[0].getElementsByTagName("Message")
                if message_nodes and message_nodes[0].firstChild:
                    event_parts.append(f"  Message: {message_nodes[0].firstChild.nodeValue}")
                
                # Get Level display name
                level_nodes = rendering_info_nodes[0].getElementsByTagName("Level")
                if level_nodes and level_nodes[0].firstChild:
                    event_parts.append(f"  Level Name: {level_nodes[0].firstChild.nodeValue}")
                
                # Get Task display name
                task_nodes = rendering_info_nodes[0].getElementsByTagName("Task")
                if task_nodes and task_nodes[0].firstChild:
                    event_parts.append(f"  Task Name: {task_nodes[0].firstChild.nodeValue}")
            
            # Check for RenderedMessage attribute (for simplified output)
            rendered_message = event_node.getAttribute("RenderedMessage")
            if rendered_message:
                event_parts.append("\nRendered Message:")
                event_parts.append(rendered_message)
            
            # Format all parts into a single string
            if event_parts:
                return "\n".join(event_parts)
            else:
                return ""
                
        except Exception as e:
            logging.error(f"Error extracting event text: {e}")
            return f"Error parsing event: {str(e)}"

def evtx_to_text(file_path: str) -> str:
    """
    Convert an EVTX file to text format for RAG processing
    
    Args:
        file_path: Path to the EVTX file
        
    Returns:
        String containing the extracted text
    """
    try:
        parser = SimpleEvtxParser(file_path)
        return parser.extract_text()
    except Exception as e:
        logging.error(f"Error converting EVTX to text: {e}")
        return f"Error processing {os.path.basename(file_path)}: {str(e)}"