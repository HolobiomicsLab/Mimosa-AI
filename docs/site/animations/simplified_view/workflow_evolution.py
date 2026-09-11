from manim import *
import numpy as np

# Color palette - professional conference style
COLORS = {
    "bg": "#0d1117",
    "solution_space": "#1a1f2e",
    "contour_low": "#1e3a5f",
    "contour_high": "#ff6b35",
    "optimal": "#00ff88",
    "workflow_node": "#4a9eff",
    "workflow_edge": "#ffffff",
    "trajectory_bad": "#ff4757",
    "trajectory_good": "#2ed573",
    "mutation": "#ffa502",
    "text": "#e6edf3",
    "subregion": ["#3742fa", "#2ed573", "#ff6b35", "#a55eea"],
}


class WorkflowEvolutionAnimation(Scene):
    def construct(self):
        self.camera.background_color = COLORS["bg"]
        
        # Scene 1: The Problem - Solution Space
        self.show_solution_space()
        self.wait(0.5)
        
        # Scene 2: Single Agent Drift
        self.show_single_agent_drift()
        self.wait(0.5)
        
        # Scene 3: Multi-Agent Partitioning
        self.show_multi_agent_partitioning()
        self.wait(0.5)
        
        # Scene 4: Evolutionary Optimization
        self.show_evolutionary_optimization()
        self.wait(0.5)
        
        # Scene 5: Final Summary
        self.show_summary()
        self.wait(1)

    def create_solution_landscape(self, axes):
        """Create a fitness landscape surface representation"""
        # Create contour-like representation of solution space
        contours = VGroup()
        
        # Multiple contour levels
        optimal_pos = np.array([1.5, 1.0])
        
        for i, radius in enumerate(np.linspace(0.3, 2.5, 8)):
            opacity = 0.3 - i * 0.03
            color = interpolate_color(
                ManimColor(COLORS["contour_high"]),
                ManimColor(COLORS["contour_low"]),
                i / 7
            )
            ellipse = Ellipse(
                width=radius * 2,
                height=radius * 1.5,
                color=color,
                fill_opacity=opacity,
                stroke_width=1.5
            )
            ellipse.move_to(axes.c2p(optimal_pos[0], optimal_pos[1]))
            ellipse.rotate(PI / 6)
            contours.add(ellipse)
        
        return contours

    def show_solution_space(self):
        """Scene 1: Introduce the solution space"""
        # Title
        title = Text("Solution Space S", font_size=42, color=COLORS["text"])
        title.to_edge(UP, buff=0.5)
        
        # Create axes
        axes = Axes(
            x_range=[-1, 4, 1],
            y_range=[-1, 3, 1],
            x_length=8,
            y_length=5,
            axis_config={"color": COLORS["text"], "stroke_width": 2},
        )
        axes.shift(DOWN * 0.3)
        
        # Solution landscape
        contours = self.create_solution_landscape(axes)
        
        # Optimal point
        optimal = Dot(axes.c2p(1.5, 1.0), color=COLORS["optimal"], radius=0.15)
        optimal_label = MathTex("s^*", font_size=36, color=COLORS["optimal"])
        optimal_label.next_to(optimal, UR, buff=0.1)
        
        # Task label
        task_label = MathTex(r"\text{Task } T", font_size=32, color=COLORS["text"])
        task_label.to_corner(UL, buff=0.8)
        
        # Equation
        equation = MathTex(
            r"\mathcal{G}^* = \arg\max_{\mathcal{G} \in \mathcal{W}} J(\mathcal{G}, T)",
            font_size=32,
            color=COLORS["text"]
        )
        equation.to_edge(DOWN, buff=0.5)
        
        # Animations
        self.play(Write(title), run_time=1)
        self.play(Create(axes), run_time=1)
        self.play(
            LaggedStart(*[FadeIn(c) for c in contours], lag_ratio=0.1),
            run_time=2
        )
        self.play(
            GrowFromCenter(optimal),
            Write(optimal_label),
            run_time=1
        )
        self.play(Write(task_label), Write(equation), run_time=1)
        
        self.wait(1)
        
        # Store for next scene
        self.axes = axes
        self.contours = contours
        self.optimal = optimal
        self.optimal_label = optimal_label
        
        # Clear for next scene
        self.play(
            FadeOut(title),
            FadeOut(task_label),
            FadeOut(equation),
            run_time=0.8
        )

    def show_single_agent_drift(self):
        """Scene 2: Show how single agent drifts from optimal"""
        # New title
        title = Text("Single Agent: Trajectory Drift", font_size=38, color=COLORS["text"])
        title.to_edge(UP, buff=0.5)
        
        # Starting point
        start = Dot(self.axes.c2p(-0.5, 2.2), color=COLORS["workflow_node"], radius=0.12)
        start_label = Text("Start", font_size=24, color=COLORS["text"])
        start_label.next_to(start, UP, buff=0.1)
        
        # Create drifting trajectory (veers away from optimal)
        drift_points = [
            [-0.5, 2.2],
            [0.2, 1.8],
            [0.6, 1.5],
            [0.9, 1.3],
            [1.1, 1.15],  # Getting close
            [1.3, 1.2],   # Then drifts
            [1.6, 1.5],
            [2.0, 1.9],
            [2.5, 2.3],   # Drifting away
            [3.0, 2.5],
        ]
        
        drift_path = VMobject(color=COLORS["trajectory_bad"], stroke_width=4)
        drift_path.set_points_smoothly([
            self.axes.c2p(p[0], p[1]) for p in drift_points
        ])
        
        # Drift annotation
        drift_text = Text("Drift from optimal", font_size=24, color=COLORS["trajectory_bad"])
        drift_text.move_to(self.axes.c2p(2.5, 2.8))
        
        # Equation showing drift
        drift_eq = MathTex(
            r"P(s \mid \mathbf{c}_t) \neq P(s \mid T)",
            font_size=28,
            color=COLORS["trajectory_bad"]
        )
        drift_eq.to_edge(DOWN, buff=0.5)
        
        # Context accumulation visualization
        context_boxes = VGroup()
        for i in range(5):
            box = Rectangle(
                width=0.4, height=0.25,
                color=COLORS["text"],
                fill_opacity=0.3 + i * 0.1,
                stroke_width=1
            )
            context_boxes.add(box)
        context_boxes.arrange(RIGHT, buff=0.05)
        context_boxes.to_corner(DR, buff=1)
        context_label = Text("Context accumulates noise", font_size=20, color=COLORS["text"])
        context_label.next_to(context_boxes, UP, buff=0.2)
        
        # Animations
        self.play(Write(title), run_time=0.8)
        self.play(GrowFromCenter(start), Write(start_label), run_time=0.5)
        
        # Animate trajectory with growing context
        self.play(
            Create(drift_path),
            LaggedStart(*[FadeIn(b) for b in context_boxes], lag_ratio=0.3),
            run_time=3
        )
        
        self.play(
            Write(drift_text),
            Write(drift_eq),
            Write(context_label),
            run_time=1
        )
        
        self.wait(1.5)
        
        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(start), FadeOut(start_label),
            FadeOut(drift_path),
            FadeOut(drift_text),
            FadeOut(drift_eq),
            FadeOut(context_boxes),
            FadeOut(context_label),
            run_time=0.8
        )

    def show_multi_agent_partitioning(self):
        """Scene 3: Show how MAS partitions the solution space"""
        # Title
        title = Text("Multi-Agent: Space Partitioning", font_size=38, color=COLORS["text"])
        title.to_edge(UP, buff=0.5)
        
        # Create partition regions
        regions = VGroup()
        region_data = [
            {"center": [0.3, 1.8], "w": 1.8, "h": 1.4, "angle": 0.2, "label": "R_1"},
            {"center": [1.5, 0.5], "w": 2.0, "h": 1.2, "angle": -0.1, "label": "R_2"},
            {"center": [2.5, 1.8], "w": 1.6, "h": 1.3, "angle": 0.15, "label": "R_3"},
        ]
        
        for i, rd in enumerate(region_data):
            region = Ellipse(
                width=rd["w"],
                height=rd["h"],
                color=COLORS["subregion"][i],
                fill_opacity=0.25,
                stroke_width=3
            )
            region.move_to(self.axes.c2p(rd["center"][0], rd["center"][1]))
            region.rotate(rd["angle"])
            
            label = MathTex(rd["label"], font_size=28, color=COLORS["subregion"][i])
            label.move_to(region.get_center())
            
            regions.add(VGroup(region, label))
        
        # Workflow graph (small, in corner)
        workflow = VGroup()
        nodes = []
        node_positions = [LEFT * 1.5, ORIGIN, RIGHT * 1.5]
        node_labels = ["a_1", "a_2", "a_3"]
        
        for i, (pos, lbl) in enumerate(zip(node_positions, node_labels)):
            node = Circle(radius=0.25, color=COLORS["subregion"][i], fill_opacity=0.8)
            node.move_to(pos)
            label = MathTex(lbl, font_size=20, color=WHITE)
            label.move_to(node.get_center())
            nodes.append(VGroup(node, label))
            workflow.add(nodes[-1])
        
        # Edges
        for i in range(2):
            edge = Arrow(
                nodes[i][0].get_right(),
                nodes[i+1][0].get_left(),
                buff=0.05,
                color=COLORS["workflow_edge"],
                stroke_width=2
            )
            workflow.add(edge)
        
        workflow.scale(0.8)
        workflow.to_corner(DR, buff=0.8)
        
        workflow_label = MathTex(r"\mathcal{G} = (V, E)", font_size=24, color=COLORS["text"])
        workflow_label.next_to(workflow, UP, buff=0.2)
        
        # Equation
        partition_eq = MathTex(
            r"s_i^* = \arg\max_{s \in R_i} Q_i(s \mid T_i)",
            font_size=28,
            color=COLORS["text"]
        )
        partition_eq.to_edge(DOWN, buff=0.5)
        
        # Local trajectories (staying within regions)
        trajectories = VGroup()
        traj_data = [
            {"region": 0, "points": [[0.0, 2.0], [0.2, 1.8], [0.35, 1.75]]},
            {"region": 1, "points": [[0.8, 0.6], [1.2, 0.5], [1.5, 0.55], [1.7, 0.5]]},
            {"region": 2, "points": [[2.2, 2.0], [2.4, 1.8], [2.5, 1.7]]},
        ]
        
        for td in traj_data:
            traj = VMobject(
                color=COLORS["trajectory_good"],
                stroke_width=3
            )
            traj.set_points_smoothly([
                self.axes.c2p(p[0], p[1]) for p in td["points"]
            ])
            trajectories.add(traj)
        
        # Animations
        self.play(Write(title), run_time=0.8)
        
        # Show regions appearing
        self.play(
            LaggedStart(*[FadeIn(r) for r in regions], lag_ratio=0.3),
            run_time=2
        )
        
        # Show workflow
        self.play(
            FadeIn(workflow),
            Write(workflow_label),
            run_time=1
        )
        
        # Show local trajectories
        self.play(
            LaggedStart(*[Create(t) for t in trajectories], lag_ratio=0.2),
            run_time=2
        )
        
        self.play(Write(partition_eq), run_time=0.8)
        
        # Highlight the problem: static topology
        problem_text = Text(
            "But: Fixed topology may be suboptimal",
            font_size=26,
            color=COLORS["mutation"]
        )
        problem_text.next_to(title, DOWN, buff=0.3)
        
        self.wait(1)
        self.play(Write(problem_text), run_time=1)
        
        self.wait(1.5)
        
        # Store regions for next scene
        self.regions = regions
        self.workflow = workflow
        self.trajectories = trajectories
        
        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(problem_text),
            FadeOut(partition_eq),
            FadeOut(workflow_label),
            FadeOut(trajectories),
            run_time=0.8
        )

    def show_evolutionary_optimization(self):
        """Scene 4: Show evolutionary search over workflow space"""
        # Title
        title = Text("Evolutionary Workflow Optimization", font_size=38, color=COLORS["text"])
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=0.8)
        
        # Show the evolution loop
        loop_steps = [
            ("1. Select", "sim(T_i, T_{new}) \\geq \\delta"),
            ("2. Mutate", "\\mathcal{G}_{child} = \\mathcal{M}(\\mathcal{G}_{parent})"),
            ("3. Evaluate", "s = \\text{Judge}(\\tau, T)"),
            ("4. Update", "\\mathcal{L} \\leftarrow \\mathcal{L} \\cup \\{...\\}"),
        ]
        
        # Create step boxes on the right
        step_group = VGroup()
        for i, (step_name, equation) in enumerate(loop_steps):
            box = RoundedRectangle(
                width=3.5, height=0.8,
                corner_radius=0.1,
                color=COLORS["workflow_node"],
                fill_opacity=0.2,
                stroke_width=2
            )
            
            name_text = Text(step_name, font_size=22, color=COLORS["text"])
            name_text.move_to(box.get_left() + RIGHT * 0.8)
            
            step_group.add(VGroup(box, name_text))
        
        step_group.arrange(DOWN, buff=0.3)
        step_group.to_edge(RIGHT, buff=0.5)
        step_group.shift(DOWN * 0.3)
        
        # Add circular arrows between steps
        arrows = VGroup()
        for i in range(3):
            arrow = Arrow(
                step_group[i].get_bottom(),
                step_group[i+1].get_top(),
                buff=0.1,
                color=COLORS["mutation"],
                stroke_width=2
            )
            arrows.add(arrow)
        
        # Loop back arrow
        loop_arrow = CurvedArrow(
            step_group[3].get_bottom() + DOWN * 0.2,
            step_group[0].get_top() + UP * 0.2,
            angle=-TAU/2,
            color=COLORS["mutation"],
            stroke_width=2
        )
        loop_arrow.shift(RIGHT * 0.3)
        
        # Animate mutation on the solution space
        # Show regions morphing
        self.play(
            LaggedStart(*[FadeIn(s) for s in step_group], lag_ratio=0.2),
            run_time=1.5
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2),
            run_time=1
        )
        self.play(Create(loop_arrow), run_time=0.8)
        
        # Now show the mutation visually
        # Morph regions to better configuration
        new_region_data = [
            {"center": [0.5, 1.5], "w": 1.5, "h": 1.2, "angle": 0.1},
            {"center": [1.5, 1.0], "w": 1.8, "h": 1.4, "angle": 0},  # Centered on optimal!
            {"center": [2.8, 1.5], "w": 1.4, "h": 1.0, "angle": -0.1},
        ]
        
        new_regions = VGroup()
        for i, rd in enumerate(new_region_data):
            region = Ellipse(
                width=rd["w"],
                height=rd["h"],
                color=COLORS["subregion"][i],
                fill_opacity=0.25,
                stroke_width=3
            )
            region.move_to(self.axes.c2p(rd["center"][0], rd["center"][1]))
            region.rotate(rd["angle"])
            
            label = MathTex(f"R_{i+1}'", font_size=28, color=COLORS["subregion"][i])
            label.move_to(region.get_center())
            
            new_regions.add(VGroup(region, label))
        
        # Mutation flash
        mutation_flash = Text("MUTATE", font_size=32, color=COLORS["mutation"])
        mutation_flash.move_to(self.axes.c2p(1.5, 2.5))
        
        self.play(
            FadeIn(mutation_flash, scale=1.5),
            run_time=0.3
        )
        self.play(
            FadeOut(mutation_flash),
            *[Transform(self.regions[i], new_regions[i]) for i in range(3)],
            run_time=1.5
        )
        
        # Show improved trajectory
        good_traj = VMobject(color=COLORS["trajectory_good"], stroke_width=4)
        good_traj.set_points_smoothly([
            self.axes.c2p(0.5, 1.5),
            self.axes.c2p(0.8, 1.3),
            self.axes.c2p(1.1, 1.1),
            self.axes.c2p(1.4, 1.0),
            self.axes.c2p(1.5, 1.0),  # Reaches optimal!
        ])
        
        self.play(Create(good_traj), run_time=1.5)
        
        # Success indicator
        check = Text("✓", font_size=48, color=COLORS["optimal"])
        check.next_to(self.optimal, RIGHT, buff=0.2)
        
        self.play(FadeIn(check, scale=1.5), run_time=0.5)
        
        # Score improvement
        score_text = VGroup(
            Text("Score: 0.67 → 0.94", font_size=24, color=COLORS["trajectory_good"])
        )
        score_text.to_corner(DL, buff=1)
        
        self.play(Write(score_text), run_time=0.8)
        
        self.wait(2)
        
        # Store for cleanup
        self.step_group = step_group
        self.arrows = arrows
        self.loop_arrow = loop_arrow
        self.good_traj = good_traj
        self.check = check
        self.score_text = score_text
        self.title = title

    def show_summary(self):
        """Scene 5: Final summary"""
        # Fade out previous elements
        self.play(
            FadeOut(self.step_group),
            FadeOut(self.arrows),
            FadeOut(self.loop_arrow),
            FadeOut(self.good_traj),
            FadeOut(self.check),
            FadeOut(self.score_text),
            FadeOut(self.title),
            FadeOut(self.axes),
            FadeOut(self.contours),
            FadeOut(self.optimal),
            FadeOut(self.optimal_label),
            FadeOut(self.regions),
            FadeOut(self.workflow),
            run_time=1
        )
        
        # Final summary screen
        final_title = Text(
            "Self-Evolving Multi-Agent Workflows",
            font_size=44,
            color=COLORS["text"]
        )
        final_title.to_edge(UP, buff=1)
        
        # Key points
        points = VGroup(
            Text("• Workflow topology constrains agent search space", font_size=28),
            Text("• Static topologies are suboptimal", font_size=28),
            Text("• Evolution discovers better constraint placement", font_size=28),
            Text("• Library enables transfer across tasks", font_size=28),
        )
        points.set_color(COLORS["text"])
        points.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        points.center()
        
        # Final equation
        final_eq = MathTex(
            r"\mathcal{G}^* = \arg\max_{\mathcal{G} \in \mathcal{W}} \mathbb{E}_{\tau}[\text{Perf}(\tau, T)]",
            font_size=36,
            color=COLORS["optimal"]
        )
        final_eq.to_edge(DOWN, buff=1)
        
        self.play(Write(final_title), run_time=1)
        self.play(
            LaggedStart(*[FadeIn(p, shift=RIGHT*0.5) for p in points], lag_ratio=0.3),
            run_time=2
        )
        self.play(Write(final_eq), run_time=1)
        
        self.wait(2)


class WorkflowEvolutionShort(Scene):
    """Shorter version focusing on the core concept"""
    def construct(self):
        self.camera.background_color = COLORS["bg"]
        
        # Title
        title = Text(
            "Finding Optimal Workflow Topology",
            font_size=40,
            color=COLORS["text"]
        )
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        
        # Split screen: Workflow space | Solution space
        
        # Left: Workflow space
        workflow_box = RoundedRectangle(
            width=5.5, height=4,
            corner_radius=0.2,
            color=COLORS["text"],
            stroke_width=2
        )
        workflow_box.shift(LEFT * 3.5)
        
        workflow_label = Text("Workflow Space W", font_size=24, color=COLORS["text"])
        workflow_label.next_to(workflow_box, UP, buff=0.2)
        
        # Right: Solution space
        solution_box = RoundedRectangle(
            width=5.5, height=4,
            corner_radius=0.2,
            color=COLORS["text"],
            stroke_width=2
        )
        solution_box.shift(RIGHT * 3.5)
        
        solution_label = Text("Solution Space S", font_size=24, color=COLORS["text"])
        solution_label.next_to(solution_box, UP, buff=0.2)
        
        self.play(
            Create(workflow_box), Write(workflow_label),
            Create(solution_box), Write(solution_label),
            run_time=1
        )
        
        # Arrow between them
        mapping_arrow = Arrow(
            workflow_box.get_right(),
            solution_box.get_left(),
            buff=0.3,
            color=COLORS["mutation"],
            stroke_width=3
        )
        mapping_label = MathTex(
            r"\mathcal{G} \mapsto \tau",
            font_size=24,
            color=COLORS["mutation"]
        )
        mapping_label.next_to(mapping_arrow, UP, buff=0.1)
        
        self.play(GrowArrow(mapping_arrow), Write(mapping_label))
        
        # Show workflow graphs on left
        workflows = []
        positions = [UP * 1 + LEFT * 3.5, LEFT * 3.5, DOWN * 1 + LEFT * 3.5]
        scores = ["0.4", "0.7", "0.9"]
        
        for i, (pos, score) in enumerate(zip(positions, scores)):
            # Mini workflow graph
            nodes = VGroup()
            for j, offset in enumerate([LEFT * 0.5, ORIGIN, RIGHT * 0.5]):
                node = Circle(
                    radius=0.15,
                    color=COLORS["subregion"][j % 4],
                    fill_opacity=0.8
                )
                node.move_to(pos + offset)
                nodes.add(node)
            
            edges = VGroup()
            for j in range(2):
                edge = Line(
                    nodes[j].get_right(),
                    nodes[j+1].get_left(),
                    color=COLORS["workflow_edge"],
                    stroke_width=1.5
                )
                edges.add(edge)
            
            score_label = Text(f"s={score}", font_size=16, color=COLORS["text"])
            score_label.next_to(nodes, RIGHT, buff=0.3)
            
            workflow = VGroup(nodes, edges, score_label)
            workflows.append(workflow)
        
        # Show trajectories on right corresponding to each workflow
        optimal_pos = RIGHT * 3.5 + UP * 0.5
        optimal_dot = Dot(optimal_pos, color=COLORS["optimal"], radius=0.15)
        optimal_label = MathTex("s^*", font_size=24, color=COLORS["optimal"])
        optimal_label.next_to(optimal_dot, UR, buff=0.05)
        
        self.play(GrowFromCenter(optimal_dot), Write(optimal_label))
        
        trajectories = []
        traj_colors = [COLORS["trajectory_bad"], COLORS["mutation"], COLORS["trajectory_good"]]
        traj_endpoints = [
            RIGHT * 3.5 + DOWN * 1 + RIGHT * 0.8,  # Bad
            RIGHT * 3.5 + UP * 0.2 + LEFT * 0.3,   # Medium
            optimal_pos,  # Good - reaches optimal
        ]
        
        for i, (workflow, color, endpoint) in enumerate(zip(workflows, traj_colors, traj_endpoints)):
            start = solution_box.get_left() + RIGHT * 0.5 + UP * (1 - i)
            
            traj = VMobject(color=color, stroke_width=3)
            traj.set_points_smoothly([start, (start + endpoint) / 2, endpoint])
            trajectories.append(traj)
            
            # Animate workflow and trajectory together
            self.play(
                FadeIn(workflow),
                Create(traj),
                run_time=1
            )
            self.wait(0.3)
        
        # Highlight the evolution
        evolution_text = Text(
            "Evolution finds better topologies",
            font_size=28,
            color=COLORS["optimal"]
        )
        evolution_text.to_edge(DOWN, buff=0.8)
        
        # Highlight best workflow
        highlight = SurroundingRectangle(
            workflows[2],
            color=COLORS["optimal"],
            stroke_width=3,
            buff=0.15
        )
        
        self.play(
            Create(highlight),
            Write(evolution_text),
            run_time=1
        )
        
        self.wait(2)


if __name__ == "__main__":
    # This allows running with: python workflow_evolution.py
    import subprocess
    subprocess.run([
        "manim", "-pqh", __file__, "WorkflowEvolutionAnimation"
    ])
