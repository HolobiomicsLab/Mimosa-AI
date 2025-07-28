
def collect_goals_from_user() -> list[str]:
    """Collect goals from user input for mass testing.
    
    Returns:
        List of goal strings entered by the user
    """
    goals = []
    print("\n🎯 Mass Testing Mode - Enter your goals")
    print("=" * 50)
    print("Enter goals one at a time. Press Enter with empty input to finish.")
    print("Type 'quit' or 'exit' to cancel.\n")
    
    goal_count = 1
    while True:
        try:
            goal = input(f"Goal {goal_count}: ").strip()
            if not goal:
                if goals:
                    break
                else:
                    print("⚠️ Please enter at least one goal or type 'quit' to cancel.")
                    continue
            if goal.lower() in ['quit', 'exit']:
                print("❌ Mass testing cancelled by user.")
                return []
            goals.append(goal)
            goal_count += 1
        except KeyboardInterrupt:
            print("\n❌ Mass testing cancelled by user.")
            return []
    print(f"\n✅ Collected {len(goals)} goals for mass testing:")
    for i, goal in enumerate(goals, 1):
        print(f"  {i}. {goal[:60]}{'...' if len(goal) > 60 else ''}")
    return goals