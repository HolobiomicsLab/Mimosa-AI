#!/usr/bin/env python3
"""
Interactive Memory Explorer for Multi-Agent Traces
Usage: python memory_explorer.py <run_uuid>
Example: python memory_explorer.py 20260115_113303_9bb63437
"""

import curses
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


class MemoryExplorer:
    def __init__(self, stdscr, memory_data: List[Dict]):
        self.stdscr = stdscr
        self.memory_data = memory_data
        self.current_step = len(memory_data) - 1  # Start at last step (contains full history)
        self.scroll_offset = 0
        self.view_mode = 'overview'  # overview, messages, full
        
        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)
        
    def get_step_data(self):
        """Get current step data"""
        return self.memory_data[self.current_step]
    
    def format_overview(self, step_data: Dict) -> List[str]:
        """Format step overview"""
        lines = []
        lines.append(f"═══ STEP {step_data.get('step', 'N/A')} / {len(self.memory_data) - 1} ═══")
        lines.append("")
        
        # Timing info
        timing = step_data.get('timing', {})
        if timing:
            duration = timing.get('duration', 0)
            lines.append(f"⏱  Duration: {duration:.2f}s")
        
        # Token usage
        token_usage = step_data.get('token_usage', {})
        if token_usage:
            lines.append(f"🔢 Tokens: {token_usage.get('input_tokens', 0)} in / {token_usage.get('output_tokens', 0)} out / {token_usage.get('total_tokens', 0)} total")
        
        lines.append("")
        
        # Error status
        error = step_data.get('error')
        if error:
            lines.append(f"❌ Error: {error}")
            lines.append("")
        
        # Action output
        action_output = step_data.get('action_output')
        if action_output:
            if isinstance(action_output, dict):
                status = action_output.get('status', 'N/A')
                message = action_output.get('message', '')
                lines.append(f"📤 Action Output:")
                lines.append(f"   Status: {status}")
                if message:
                    lines.append(f"   Message: {message}")
            else:
                lines.append(f"📤 Action Output: {action_output}")
            lines.append("")
        
        # Message count
        messages = step_data.get('model_input_messages', [])
        lines.append(f"💬 Messages in history: {len(messages)}")
        
        # Tool calls
        tool_calls = step_data.get('tool_calls', [])
        if tool_calls:
            lines.append(f"🔧 Tool calls: {len(tool_calls)}")
            for i, tc in enumerate(tool_calls):
                func_name = tc.get('function', {}).get('name', 'unknown')
                lines.append(f"   {i+1}. {func_name}")
        
        lines.append("")
        lines.append("─" * 60)
        lines.append("📝 Model Output Preview:")
        lines.append("─" * 60)
        model_output = step_data.get('model_output', '')
        if model_output:
            preview_lines = model_output.split('\n')[:10]
            lines.extend(preview_lines)
            if len(model_output.split('\n')) > 10:
                lines.append("... (press 'm' to see full messages)")
        
        return lines
    
    def format_messages(self, step_data: Dict) -> List[str]:
        """Format message history"""
        lines = []
        lines.append(f"═══ MESSAGE HISTORY - STEP {step_data.get('step', 'N/A')} ═══")
        lines.append("")
        
        messages = step_data.get('model_input_messages', [])
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', [])
            
            lines.append(f"┌─ Message {i+1} [{role.upper()}] ─")
            
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        text = c.get('text', '')
                        if text:
                            # Truncate long messages
                            text_lines = text.split('\n')
                            for line in text_lines[:50]:  # Show first 50 lines
                                lines.append(f"│ {line}")
                            if len(text_lines) > 50:
                                lines.append(f"│ ... ({len(text_lines) - 50} more lines)")
            elif isinstance(content, str):
                text_lines = content.split('\n')
                for line in text_lines[:50]:
                    lines.append(f"│ {line}")
                if len(text_lines) > 50:
                    lines.append(f"│ ... ({len(text_lines) - 50} more lines)")
            
            lines.append("└" + "─" * 70)
            lines.append("")
        
        return lines
    
    def format_full(self, step_data: Dict) -> List[str]:
        """Format full JSON"""
        lines = []
        lines.append(f"═══ FULL JSON - STEP {step_data.get('step', 'N/A')} ═══")
        lines.append("")
        json_str = json.dumps(step_data, indent=2)
        lines.extend(json_str.split('\n'))
        return lines
    
    def draw_help_bar(self, height: int, width: int):
        """Draw help bar at bottom"""
        help_text = "↑↓:Scroll  ←→:Steps  o:Overview  m:Messages  f:Full  q:Quit"
        try:
            self.stdscr.attron(curses.color_pair(6))
            self.stdscr.addstr(height - 1, 0, help_text.ljust(width)[:width])
            self.stdscr.attroff(curses.color_pair(6))
        except:
            pass
    
    def draw_header(self, width: int):
        """Draw header"""
        header = f"📊 Memory Explorer - Step {self.current_step + 1}/{len(self.memory_data)} - Mode: {self.view_mode.upper()}"
        try:
            self.stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
            self.stdscr.addstr(0, 0, header.ljust(width)[:width])
            self.stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
        except:
            pass
    
    def draw_content(self, lines: List[str], start_line: int, height: int, width: int):
        """Draw content area"""
        viewable_height = height - 3  # Leave room for header and help bar
        
        for i in range(viewable_height):
            line_idx = self.scroll_offset + i
            if line_idx < len(lines):
                line = lines[line_idx]
                try:
                    # Apply colors based on content
                    if line.startswith('═══'):
                        self.stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                        self.stdscr.addstr(start_line + i, 0, line[:width])
                        self.stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
                    elif line.startswith('❌'):
                        self.stdscr.attron(curses.color_pair(4))
                        self.stdscr.addstr(start_line + i, 0, line[:width])
                        self.stdscr.attroff(curses.color_pair(4))
                    elif line.startswith('📤') or line.startswith('🔧') or line.startswith('💬'):
                        self.stdscr.attron(curses.color_pair(2))
                        self.stdscr.addstr(start_line + i, 0, line[:width])
                        self.stdscr.attroff(curses.color_pair(2))
                    elif line.startswith('┌─') or line.startswith('['):
                        self.stdscr.attron(curses.color_pair(3))
                        self.stdscr.addstr(start_line + i, 0, line[:width])
                        self.stdscr.attroff(curses.color_pair(3))
                    else:
                        self.stdscr.addstr(start_line + i, 0, line[:width])
                except:
                    pass
    
    def run(self):
        """Main loop"""
        self.stdscr.clear()
        curses.curs_set(0)  # Hide cursor
        
        while True:
            height, width = self.stdscr.getmaxyx()
            self.stdscr.clear()
            
            # Get content based on view mode
            step_data = self.get_step_data()
            if self.view_mode == 'overview':
                lines = self.format_overview(step_data)
            elif self.view_mode == 'messages':
                lines = self.format_messages(step_data)
            elif self.view_mode == 'full':
                lines = self.format_full(step_data)
            else:
                lines = ["Unknown view mode"]
            
            # Draw UI
            self.draw_header(width)
            self.draw_content(lines, 1, height, width)
            self.draw_help_bar(height, width)
            
            # Show scroll indicator
            if len(lines) > height - 3:
                scroll_pct = int((self.scroll_offset / max(1, len(lines) - height + 3)) * 100)
                try:
                    self.stdscr.addstr(height - 2, width - 10, f"[{scroll_pct}%]", curses.color_pair(5))
                except:
                    pass
            
            self.stdscr.refresh()
            
            # Handle input
            key = self.stdscr.getch()
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == curses.KEY_UP:
                self.scroll_offset = max(0, self.scroll_offset - 1)
            elif key == curses.KEY_DOWN:
                max_scroll = max(0, len(lines) - (height - 3))
                self.scroll_offset = min(max_scroll, self.scroll_offset + 1)
            elif key == curses.KEY_LEFT:
                if self.current_step > 0:
                    self.current_step -= 1
                    self.scroll_offset = 0
            elif key == curses.KEY_RIGHT:
                if self.current_step < len(self.memory_data) - 1:
                    self.current_step += 1
                    self.scroll_offset = 0
            elif key == ord('o') or key == ord('O'):
                self.view_mode = 'overview'
                self.scroll_offset = 0
            elif key == ord('m') or key == ord('M'):
                self.view_mode = 'messages'
                self.scroll_offset = 0
            elif key == ord('f') or key == ord('F'):
                self.view_mode = 'full'
                self.scroll_offset = 0
            elif key == curses.KEY_HOME:
                self.scroll_offset = 0
            elif key == curses.KEY_END:
                max_scroll = max(0, len(lines) - (height - 3))
                self.scroll_offset = max_scroll
            elif key == curses.KEY_PPAGE:  # Page Up
                self.scroll_offset = max(0, self.scroll_offset - (height - 3))
            elif key == curses.KEY_NPAGE:  # Page Down
                max_scroll = max(0, len(lines) - (height - 3))
                self.scroll_offset = min(max_scroll, self.scroll_offset + (height - 3))


def select_json_file(memory_dir: Path) -> Path:
    """Interactively select a JSON file"""
    json_files = sorted(list(memory_dir.glob("*.json")))
    
    if not json_files:
        print(f"❌ No JSON files found in {memory_dir}")
        sys.exit(1)
    
    print(f"\n📂 Found {len(json_files)} memory trace(s) in {memory_dir.name}:\n")
    for i, f in enumerate(json_files, 1):
        size = f.stat().st_size
        size_kb = size / 1024
        print(f"  {i}. {f.name} ({size_kb:.1f} KB)")
    
    while True:
        try:
            choice = input(f"\n🔍 Select file (1-{len(json_files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(json_files):
                return json_files[idx]
            else:
                print(f"⚠️  Please enter a number between 1 and {len(json_files)}")
        except ValueError:
            print("⚠️  Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Cancelled")
            sys.exit(0)


def main():
    if len(sys.argv) != 2:
        print("Usage: python memory_explorer.py <run_uuid>")
        print("Example: python memory_explorer.py 20260115_113303_9bb63437")
        sys.exit(1)
    
    run_uuid = sys.argv[1]
    memory_dir = Path("sources/memory") / run_uuid
    
    if not memory_dir.exists():
        print(f"❌ Memory directory not found: {memory_dir}")
        sys.exit(1)
    
    # Select JSON file
    json_file = select_json_file(memory_dir)
    
    print(f"\n📖 Loading {json_file.name}...")
    
    # Load JSON
    try:
        with open(json_file, 'r') as f:
            memory_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        sys.exit(1)
    
    if not isinstance(memory_data, list):
        print("❌ Expected JSON to be a list of steps")
        sys.exit(1)
    
    if len(memory_data) == 0:
        print("❌ No steps found in memory trace")
        sys.exit(1)
    
    print(f"✅ Loaded {len(memory_data)} step(s)")
    print("\n🚀 Starting interactive explorer...\n")
    
    # Start curses interface
    try:
        curses.wrapper(lambda stdscr: MemoryExplorer(stdscr, memory_data).run())
    except KeyboardInterrupt:
        pass
    
    print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()
