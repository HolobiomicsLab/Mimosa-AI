#!/usr/bin/env python3
"""
Memory Timelapse Generator
Creates a video visualization of multi-agent system traces.

Usage:
    python memory_timelapse.py --output video.mp4 --fps 2
    python memory_timelapse.py --stage stage4 --agent task_bioactivity_correlator --output bioactivity_trace.mp4
"""

import json
import os
import argparse
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio

thought_default=""

# Color scheme
COLORS = {
    'bg': (15, 23, 42),           # Dark slate background
    'bg_light': (30, 41, 59),     # Lighter background
    'accent': (99, 102, 241),     # Indigo accent
    'success': (34, 197, 94),     # Green
    'warning': (234, 179, 8),     # Yellow
    'error': (239, 68, 68),       # Red
    'info': (59, 130, 246),       # Blue
    'text': (248, 250, 252),      # White text
    'text_dim': (148, 163, 184),  # Dimmed text
    'border': (71, 85, 105),      # Border color
    'stage_colors': {
        'plan_creator': (139, 92, 246),  # Purple
        'stage2': (236, 72, 153),        # Pink
        'stage3': (14, 165, 233),        # Sky
        'stage4': (245, 158, 11),        # Amber
    }
}


@dataclass
class StepInfo:
    step: int
    stage: str
    agent: str
    duration: float
    start_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_calls: List[str]
    has_error: bool
    action_status: Optional[str]
    thought_preview: str
    code_preview: str
    observation_preview: str


def load_memory_files(memory_dir: Path) -> Dict[str, List[Dict]]:
    """Load all memory JSON files organized by stage."""
    stages = {}
    
    # Load plan_creator
    plan_file = memory_dir / "plan_creator.json"
    if plan_file.exists():
        with open(plan_file) as f:
            data = json.load(f)
            # plan_creator is a single response, not a list of steps
            stages['plan_creator'] = [{
                'step': 0,
                'stage': 'plan_creator',
                'agent': 'plan_creator',
                'timing': {'duration': 0},
                'token_usage': data.get('usage', {}),
                'tool_calls': [],
                'error': None,
                'model_output': data.get('response', '')[:500],
                'action_output': {'status': 'SUCCESS', 'message': 'Plan created'}
            }]
    
    # Load stage files
    for stage_dir in sorted(memory_dir.glob("stage*")):
        stage_name = stage_dir.name
        stages[stage_name] = []
        
        for json_file in sorted(stage_dir.glob("*.json")):
            agent_name = json_file.stem
            with open(json_file) as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    # It's a trace of steps
                    for step_data in data:
                        step_data['stage'] = stage_name
                        step_data['agent'] = agent_name
                        stages[stage_name].append(step_data)
                else:
                    # Single response
                    stages[stage_name].append({
                        'step': 0,
                        'stage': stage_name,
                        'agent': agent_name,
                        'timing': {'duration': 0},
                        'token_usage': data.get('usage', {}),
                        'tool_calls': [],
                        'error': None,
                        'model_output': data.get('response', '')[:500],
                        'action_output': {'status': 'SUCCESS', 'message': 'Task completed'}
                    })
    
    return stages


def parse_step(step_data: Dict) -> StepInfo:
    """Parse step data into structured format."""
    timing = step_data.get('timing', {})
    token_usage = step_data.get('token_usage', {})
    
    # Extract tool calls
    tool_calls = step_data.get('tool_calls', [])
    tool_names = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            func = tc.get('function', {})
            if isinstance(func, dict):
                tool_names.append(func.get('name', 'unknown'))
            else:
                tool_names.append(str(tc.get('name', 'unknown')))
    
    # Get action status
    action_output = step_data.get('action_output', {})
    if isinstance(action_output, dict):
        action_status = action_output.get('status')
    else:
        action_status = str(action_output) if action_output else None
    
    # Extract thought/code/observation previews
    model_output = step_data.get('model_output', '')
    thought_preview = ""
    code_preview = ""
    observation_preview = ""
    
    if model_output:
        lines = model_output.split('\n')
        for i, line in enumerate(lines):
            if 'Thought:' in line:
                thought_preview = lines[i + 1][:256]
                thought_default = thought_preview
            if '```py' in line or '```python' in line:
                # Find code block - extract up to 15 lines for better preview
                code_lines = []
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == '```':
                        break
                    code_lines.append(lines[j][:110])
                    if len(code_lines) >= 15:  # Get more lines for the preview
                        break
                code_preview = '\n'.join(code_lines)
    
    observations = step_data.get('observations', '')
    if observations:
        observation_preview = str(observations)[:100]
    
    return StepInfo(
        step=step_data.get('step', 0),
        stage=step_data.get('stage', 'unknown'),
        agent=step_data.get('agent', 'unknown'),
        duration=timing.get('duration', 0) if isinstance(timing, dict) else 0,
        start_time=timing.get('start_time', 0) if isinstance(timing, dict) else 0,
        input_tokens=token_usage.get('input_tokens', 0) if isinstance(token_usage, dict) else 0,
        output_tokens=token_usage.get('output_tokens', 0) if isinstance(token_usage, dict) else 0,
        total_tokens=token_usage.get('total_tokens', 0) if isinstance(token_usage, dict) else 0,
        tool_calls=tool_names,
        has_error=step_data.get('error') is not None,
        action_status=action_status,
        thought_preview=thought_preview,
        code_preview=code_preview,
        observation_preview=observation_preview
    )


def create_gradient_background(width: int, height: int, color1: Tuple, color2: Tuple) -> Image.Image:
    """Create a gradient background."""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    return img


def draw_rounded_rect(draw: ImageDraw.Draw, xy: Tuple[int, int, int, int], 
                       radius: int, fill: Tuple, outline: Tuple = None, width: int = 1):
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def format_number(n: int) -> str:
    """Format large numbers."""
    if n >= 1000000:
        return f"{n/1000000:.1f}M"
    elif n >= 1000:
        return f"{n/1000:.1f}K"
    return str(n)


def create_frame(step: StepInfo, all_steps: List[StepInfo], frame_idx: int, 
                 total_frames: int, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a single video frame for a step."""
    
    # Try to load fonts
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        font_header = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font_title = ImageFont.load_default()
        font_header = font_title
        font_body = font_title
        font_small = font_title
    
    # Create background
    img = create_gradient_background(width, height, COLORS['bg'], COLORS['bg_light'])
    draw = ImageDraw.Draw(img)
    
    # Header bar
    draw.rectangle([(0, 0), (width, 70)], fill=COLORS['bg'])
    draw.line([(0, 70), (width, 70)], fill=COLORS['border'], width=2)
    
    # Title
    title = "🤖 Multi-Agent System Trace Visualization"
    draw.text((30, 20), title, fill=COLORS['text'], font=font_title)
    
    # Progress bar
    progress = (frame_idx + 1) / total_frames
    bar_width = 300
    draw.rounded_rectangle([(width - bar_width - 30, 25), (width - 30, 45)], 
                          radius=10, fill=COLORS['bg_light'], outline=COLORS['border'])
    draw.rounded_rectangle([(width - bar_width - 30, 25), 
                           (width - bar_width - 30 + int(bar_width * progress), 45)], 
                          radius=10, fill=COLORS['accent'])
    progress_text = f"{frame_idx + 1}/{total_frames}"
    draw.text((width - 150, 28), progress_text, fill=COLORS['text'], font=font_small)
    
    # Stage indicator at top
    stage_y = 85
    stage_x = 30
    stage_spacing = 180
    stages = ['plan_creator', 'stage2', 'stage3', 'stage4']
    stage_labels = ['📋 Plan', '🔬 Stage 2', '📊 Stage 3', '🧬 Stage 4']
    
    current_stage_idx = stages.index(step.stage) if step.stage in stages else -1
    
    for i, (stage, label) in enumerate(zip(stages, stage_labels)):
        x = stage_x + i * stage_spacing
        if i < current_stage_idx:
            # Completed stage
            color = COLORS['success']
            bg_color = (20, 60, 30)
        elif i == current_stage_idx:
            # Current stage
            color = COLORS['stage_colors'].get(stage, COLORS['accent'])
            bg_color = (color[0]//4, color[1]//4, color[2]//4)
        else:
            # Future stage
            color = COLORS['text_dim']
            bg_color = COLORS['bg_light']
        
        draw.rounded_rectangle([(x, stage_y), (x + 160, stage_y + 40)], 
                              radius=8, fill=bg_color, outline=color, width=2)
        draw.text((x + 15, stage_y + 10), label, fill=color, font=font_body)
    
    # Main content area
    content_y = 140
    
    # Left panel: Step info
    left_x = 30
    left_width = 550
    
    # Agent card
    card_y = content_y
    stage_color = COLORS['stage_colors'].get(step.stage, COLORS['accent'])
    
    # Card background
    draw.rounded_rectangle([(left_x, card_y), (left_x + left_width, card_y + 400)], 
                          radius=12, fill=COLORS['bg_light'], outline=COLORS['border'], width=1)
    
    # Stage header
    draw.rounded_rectangle([(left_x, card_y), (left_x + left_width, card_y + 50)], 
                          radius=12, fill=stage_color)
    draw.text((left_x + 20, card_y + 12), f"{step.stage.upper()} • {step.agent}", 
             fill=COLORS['text'], font=font_header)
    
    # Step number
    step_text = f"Step {step.step}"
    draw.text((left_x + 20, card_y + 70), step_text, fill=COLORS['text'], font=font_header)
    
    # Duration
    duration_color = COLORS['success'] if step.duration < 10 else COLORS['warning'] if step.duration < 30 else COLORS['error']
    draw.text((left_x + 20, card_y + 110), "⏱ Duration:", fill=COLORS['text_dim'], font=font_body)
    draw.text((left_x + 140, card_y + 110), f"{step.duration:.2f}s", fill=duration_color, font=font_body)
    
    # Tokens
    draw.text((left_x + 20, card_y + 140), "🔢 Tokens:", fill=COLORS['text_dim'], font=font_body)
    draw.text((left_x + 140, card_y + 140), 
             f"In: {format_number(step.input_tokens)}  Out: {format_number(step.output_tokens)}", 
             fill=COLORS['text'], font=font_body)
    draw.text((left_x + 340, card_y + 140), f"Total: {format_number(step.total_tokens)}", 
             fill=COLORS['accent'], font=font_body)
    
    # Tool calls
    draw.text((left_x + 20, card_y + 180), "🔧 Tools:", fill=COLORS['text_dim'], font=font_body)
    if step.tool_calls:
        for i, tool in enumerate(step.tool_calls[:4]):
            tool_y = card_y + 210 + i * 28
            draw.rounded_rectangle([(left_x + 20, tool_y), (left_x + 200, tool_y + 22)], 
                                  radius=4, fill=(step_color := (40, 50, 70)))
            draw.text((left_x + 30, tool_y + 2), tool[:20], fill=COLORS['info'], font=font_small)
    else:
        draw.text((left_x + 20, card_y + 210), "No tool calls", fill=COLORS['text_dim'], font=font_body)
    
    # Status
    if step.action_status:
        status_color = COLORS['success'] if step.action_status == 'SUCCESS' else \
                      COLORS['warning'] if step.action_status == 'RETRY' else COLORS['error']
        draw.text((left_x + 20, card_y + 330), f"Status: {step.action_status}", 
                 fill=status_color, font=font_header)
    
    if step.has_error:
        draw.rounded_rectangle([(left_x + 20, card_y + 360), (left_x + 200, card_y + 390)], 
                              radius=6, fill=(60, 20, 20))
        draw.text((left_x + 35, card_y + 368), "❌ ERROR OCCURRED", fill=COLORS['error'], font=font_body)
    
    # Right panel: Activity preview
    right_x = left_x + left_width + 30
    right_width = width - right_x - 30
    
    # Thought section
    section_y = content_y
    draw.rounded_rectangle([(right_x, section_y), (right_x + right_width, section_y + 130)], 
                          radius=12, fill=COLORS['bg_light'], outline=COLORS['border'], width=1)
    draw.text((right_x + 15, section_y + 10), "💭 Thought", fill=COLORS['accent'], font=font_header)
    
    if step.thought_preview:
        wrapped = textwrap.wrap(step.thought_preview or thought_default, width=90)
        for i, line in enumerate(wrapped[:32]):
            draw.text((right_x + 15, section_y + 45 + i * 20), line, 
                     fill=COLORS['text'], font=font_small)
    else:
        draw.text((right_x + 15, section_y + 45), "Processing...", 
                 fill=COLORS['text_dim'], font=font_body)
    
    # Code section - taller to show more code
    section_y = content_y + 145
    code_section_height = 320
    draw.rounded_rectangle([(right_x, section_y), (right_x + right_width, section_y + code_section_height)], 
                          radius=12, fill=(25, 35, 55), outline=COLORS['border'], width=1)
    draw.text((right_x + 15, section_y + 10), "💻 Code", fill=COLORS['success'], font=font_header)
    
    if step.code_preview:
        # Show up to 12 lines of code
        code_lines = step.code_preview.split('\n')[:12]
        for i, line in enumerate(code_lines):
            # Truncate long lines and show with line numbers
            display_line = line[:110]
            draw.text((right_x + 15, section_y + 40 + i * 22), display_line, 
                     fill=(180, 200, 220), font=font_small)
        # Show ellipsis if there are more lines
        if len(step.code_preview.split('\n')) > 12:
            draw.text((right_x + 15, section_y + 40 + 12 * 22), "... (more code)", 
                     fill=COLORS['text_dim'], font=font_small)
    else:
        draw.text((right_x + 15, section_y + 45), "No code in this step", 
                 fill=COLORS['text_dim'], font=font_body)
    
    # Observation section - adjusted position
    section_y = content_y + 145 + code_section_height + 15
    obs_height = height - section_y - 30
    draw.rounded_rectangle([(right_x, section_y), (right_x + right_width, section_y + obs_height)], 
                          radius=12, fill=(35, 45, 35), outline=COLORS['border'], width=1)
    draw.text((right_x + 15, section_y + 10), "👁 Observation", fill=COLORS['warning'], font=font_header)
    
    if step.observation_preview:
        # Try to extract key info
        obs_text = step.observation_preview
        if len(obs_text) > 200:
            obs_text = obs_text[:200] + "..."
        wrapped = textwrap.wrap(obs_text, width=90)
        for i, line in enumerate(wrapped[:6]):
            draw.text((right_x + 15, section_y + 45 + i * 20), line, 
                     fill=(200, 220, 200), font=font_small)
    else:
        draw.text((right_x + 15, section_y + 45), "Waiting for observation...", 
                 fill=COLORS['text_dim'], font=font_body)
    
    # Timeline at bottom
    timeline_y = height - 80
    timeline_height = 60
    draw.rectangle([(0, timeline_y), (width, height)], fill=COLORS['bg'])
    draw.line([(0, timeline_y), (width, timeline_y)], fill=COLORS['border'], width=2)
    
    # Timeline label
    draw.text((30, timeline_y + 10), "Timeline", fill=COLORS['text_dim'], font=font_body)
    
    # Timeline bar
    tl_x = 120
    tl_y = timeline_y + 15
    tl_width = width - tl_x - 30
    tl_height = 20
    
    draw.rounded_rectangle([(tl_x, tl_y), (tl_x + tl_width, tl_y + tl_height)], 
                          radius=10, fill=COLORS['bg_light'])
    
    # Current position marker
    if total_frames > 1:
        pos_x = tl_x + int((frame_idx / (total_frames - 1)) * tl_width)
    else:
        pos_x = tl_x
    
    # Draw all steps on timeline
    step_positions = []
    for i, s in enumerate(all_steps):
        if len(all_steps) > 1:
            sx = tl_x + int((i / (len(all_steps) - 1)) * tl_width)
        else:
            sx = tl_x
        step_positions.append(sx)
        
        # Color based on stage
        sc = COLORS['stage_colors'].get(s.stage, COLORS['accent'])
        if s.has_error:
            sc = COLORS['error']
        
        # Draw step marker
        marker_size = 6 if i == frame_idx else 4
        draw.ellipse([(sx - marker_size, tl_y + tl_height//2 - marker_size), 
                     (sx + marker_size, tl_y + tl_height//2 + marker_size)], fill=sc)
    
    # Current position highlight
    draw.ellipse([(pos_x - 10, tl_y + tl_height//2 - 10), 
                 (pos_x + 10, tl_y + tl_height//2 + 10)], 
                outline=COLORS['accent'], width=3)
    
    # Stats summary
    total_time = sum(s.duration for s in all_steps[:frame_idx+1])
    total_tokens = sum(s.total_tokens for s in all_steps[:frame_idx+1])
    draw.text((30, timeline_y + 45), 
             f"Elapsed: {total_time:.1f}s | Tokens: {format_number(total_tokens)}", 
             fill=COLORS['text_dim'], font=font_small)
    
    return np.array(img)


def create_summary_frame(all_steps: List[StepInfo], width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a summary frame showing overall statistics."""
    
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        font_header = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font_title = ImageFont.load_default()
        font_header = font_title
        font_body = font_title
        font_small = font_title
    
    img = create_gradient_background(width, height, COLORS['bg'], COLORS['bg_light'])
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((width//2 - 400, 50), "📊 Execution Summary", fill=COLORS['text'], font=font_title)
    
    # Calculate statistics
    total_steps = len(all_steps)
    total_duration = sum(s.duration for s in all_steps)
    total_tokens = sum(s.total_tokens for s in all_steps)
    total_input = sum(s.input_tokens for s in all_steps)
    total_output = sum(s.output_tokens for s in all_steps)
    error_count = sum(1 for s in all_steps if s.has_error)
    tool_calls = sum(len(s.tool_calls) for s in all_steps)
    
    # Stage breakdown
    stage_stats = {}
    for s in all_steps:
        if s.stage not in stage_stats:
            stage_stats[s.stage] = {'steps': 0, 'tokens': 0, 'duration': 0}
        stage_stats[s.stage]['steps'] += 1
        stage_stats[s.stage]['tokens'] += s.total_tokens
        stage_stats[s.stage]['duration'] += s.duration
    
    # Main stats cards
    card_y = 150
    card_height = 120
    stats = [
        ("Total Steps", str(total_steps), COLORS['accent']),
        ("Duration", f"{total_duration:.1f}s", COLORS['success']),
        ("Total Tokens", format_number(total_tokens), COLORS['warning']),
        ("Tool Calls", str(tool_calls), COLORS['info']),
    ]
    
    card_width = (width - 100) // 4
    for i, (label, value, color) in enumerate(stats):
        x = 30 + i * (card_width + 20)
        draw.rounded_rectangle([(x, card_y), (x + card_width, card_y + card_height)], 
                              radius=12, fill=COLORS['bg_light'], outline=color, width=2)
        draw.text((x + 20, card_y + 20), label, fill=COLORS['text_dim'], font=font_body)
        draw.text((x + 20, card_y + 60), value, fill=color, font=font_header)
    
    # Stage breakdown
    stage_y = card_y + card_height + 50
    draw.text((30, stage_y), "Stage Breakdown", fill=COLORS['text'], font=font_header)
    
    stage_y += 50
    stage_height = 80
    for i, (stage, stat) in enumerate(sorted(stage_stats.items())):
        x = 30
        color = COLORS['stage_colors'].get(stage, COLORS['accent'])
        
        draw.rounded_rectangle([(x, stage_y + i * 100), (width - 30, stage_y + i * 100 + stage_height)], 
                              radius=10, fill=COLORS['bg_light'], outline=color, width=1)
        
        # Stage name
        draw.text((x + 20, stage_y + i * 100 + 20), stage.upper(), fill=color, font=font_body)
        
        # Stats
        stats_text = f"Steps: {stat['steps']} | Tokens: {format_number(stat['tokens'])} | Time: {stat['duration']:.1f}s"
        draw.text((x + 250, stage_y + i * 100 + 25), stats_text, fill=COLORS['text'], font=font_body)
        
        # Progress bar
        bar_width = 400
        max_tokens = max(s['tokens'] for s in stage_stats.values())
        if max_tokens > 0:
            progress = stat['tokens'] / max_tokens
        else:
            progress = 0
        draw.rounded_rectangle([(width - bar_width - 50, stage_y + i * 100 + 30), 
                               (width - 50, stage_y + i * 100 + 50)], 
                              radius=5, fill=COLORS['bg'])
        draw.rounded_rectangle([(width - bar_width - 50, stage_y + i * 100 + 30), 
                               (width - bar_width - 50 + int(bar_width * progress), stage_y + i * 100 + 50)], 
                              radius=5, fill=color)
    
    # Token breakdown pie chart (simplified as bar)
    pie_y = stage_y + len(stage_stats) * 100 + 50
    draw.text((30, pie_y), "Token Distribution (Input vs Output)", fill=COLORS['text'], font=font_header)
    
    pie_y += 50
    bar_width = width - 100
    total = total_input + total_output
    if total > 0:
        input_ratio = total_input / total
        output_ratio = total_output / total
    else:
        input_ratio = output_ratio = 0
    
    draw.rounded_rectangle([(30, pie_y), (30 + bar_width, pie_y + 40)], 
                          radius=10, fill=COLORS['bg_light'])
    draw.rounded_rectangle([(30, pie_y), (30 + int(bar_width * input_ratio), pie_y + 40)], 
                          radius=10, fill=COLORS['info'])
    
    draw.text((50, pie_y + 50), f"Input: {format_number(total_input)} ({input_ratio*100:.1f}%)", 
             fill=COLORS['info'], font=font_body)
    draw.text((width//2, pie_y + 50), f"Output: {format_number(total_output)} ({output_ratio*100:.1f}%)", 
             fill=COLORS['success'], font=font_body)
    
    # Footer
    draw.text((width//2 - 200, height - 50), "Multi-Agent System Execution Complete", 
             fill=COLORS['text_dim'], font=font_body)
    
    return np.array(img)


def create_timelapse(memory_dir: Path, output_file: str, fps: int = 2,
                     stage_filter: Optional[str] = None, 
                     agent_filter: Optional[str] = None,
                     width: int = 1920, height: int = 1080):
    """Create the timelapse video."""
    
    print(f"📂 Loading memory files from {memory_dir}...")
    stages = load_memory_files(memory_dir)
    
    # Collect all steps
    all_steps = []
    for stage_name, steps_data in stages.items():
        if stage_filter and stage_name != stage_filter:
            continue
        
        for step_data in steps_data:
            step = parse_step(step_data)
            if agent_filter and step.agent != agent_filter:
                continue
            all_steps.append(step)
    
    # Sort by actual execution start time (chronological order)
    all_steps.sort(key=lambda s: s.start_time)
    
    if not all_steps:
        print("❌ No steps found matching the filters")
        return
    
    print(f"✅ Loaded {len(all_steps)} steps")
    
    # Create frames
    print("🎬 Generating frames...")
    frames = []
    
    # Opening frame
    opening = create_summary_frame(all_steps, width, height)
    # Show opening for 2 seconds
    for _ in range(fps * 2):
        frames.append(opening)
    
    # Step frames
    for i, step in enumerate(all_steps):
        if (i + 1) % 10 == 0:
            print(f"  Frame {i + 1}/{len(all_steps)}...")
        frame = create_frame(step, all_steps, i, len(all_steps), width, height)
        frames.append(frame)
    
    # Closing frame (summary again)
    closing = create_summary_frame(all_steps, width, height)
    for _ in range(fps * 3):
        frames.append(closing)
    
    # Write video
    print(f"💾 Writing video to {output_file}...")
    
    # Convert frames to uint8
    frames = [f.astype(np.uint8) for f in frames]
    
    # Write using imageio
    iio.imwrite(output_file, frames, fps=fps, codec='libx264', quality=8)
    
    print(f"✅ Done! Created {output_file}")
    print(f"   Duration: {len(frames)/fps:.1f}s")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")


def save_single_frame(memory_dir: Path, frame_idx: int, output_file: str,
                     stage_filter: Optional[str] = None,
                     agent_filter: Optional[str] = None,
                     width: int = 1920, height: int = 1080):
    """Save a single frame as an image."""
    print(f"📂 Loading memory files from {memory_dir}...")
    stages = load_memory_files(memory_dir)
    
    # Collect all steps
    all_steps = []
    for stage_name, steps_data in stages.items():
        if stage_filter and stage_name != stage_filter:
            continue
        
        for step_data in steps_data:
            step = parse_step(step_data)
            if agent_filter and step.agent != agent_filter:
                continue
            all_steps.append(step)
    
    # Sort by actual execution start time (chronological order)
    all_steps.sort(key=lambda s: s.start_time)
    
    if not all_steps:
        print("❌ No steps found matching the filters")
        return
    
    if frame_idx < 0 or frame_idx >= len(all_steps):
        print(f"❌ Frame index {frame_idx} out of range (0-{len(all_steps)-1})")
        return
    
    step = all_steps[frame_idx]
    print(f"🖼️  Generating frame {frame_idx} for step {step.step} of {step.agent}...")
    
    frame = create_frame(step, all_steps, frame_idx, len(all_steps), width, height)
    img = Image.fromarray(frame.astype(np.uint8))
    img.save(output_file)
    
    print(f"✅ Saved frame to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create a timelapse video of multi-agent traces')
    parser.add_argument('--memory-dir', type=Path, default=Path('sources/memory'),
                       help='Directory containing memory files')
    parser.add_argument('--output', '-o', type=str, default='memory_timelapse.mp4',
                       help='Output video file')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second (higher = faster video)')
    parser.add_argument('--stage', type=str,
                       help='Filter by stage (plan_creator, stage2, stage3, stage4)')
    parser.add_argument('--agent', type=str,
                       help='Filter by agent name')
    parser.add_argument('--width', type=int, default=1920,
                       help='Video width')
    parser.add_argument('--height', type=int, default=1080,
                       help='Video height')
    parser.add_argument('--frame', type=int,
                       help='Save a single frame as image (specify frame index)')
    parser.add_argument('--output-image', type=str,
                       help='Output image file (for single frame mode)')
    
    args = parser.parse_args()
    
    if not args.memory_dir.exists():
        print(f"❌ Memory directory not found: {args.memory_dir}")
        return
    
    # Single frame mode
    if args.frame is not None:
        output = args.output_image or f"frame_{args.frame:04d}.png"
        save_single_frame(
            memory_dir=args.memory_dir,
            frame_idx=args.frame,
            output_file=output,
            stage_filter=args.stage,
            agent_filter=args.agent,
            width=args.width,
            height=args.height
        )
        return
    
    # Video mode
    create_timelapse(
        memory_dir=args.memory_dir,
        output_file=args.output,
        fps=args.fps,
        stage_filter=args.stage,
        agent_filter=args.agent,
        width=args.width,
        height=args.height
    )


if __name__ == '__main__':
    main()
