# Mass Testing Feature

The Mass Testing feature allows you to run multiple DGM (Darwin Godel Machine) instances in parallel to test different goals simultaneously. This is useful for evaluating the performance of the AI agent across multiple tasks and comparing results.

## Features

- **Parallel Execution**: Run multiple DGM instances simultaneously using multiprocessing
- **Automatic Results Saving**: Individual results and summary statistics are automatically saved
- **Progress Tracking**: Real-time progress updates for each running process
- **Error Handling**: Robust error handling with detailed error reporting
- **Configurable Workers**: Control the number of parallel processes

## Usage

### Basic Usage

```bash
python main.py --multi_goal
```

This will prompt you to enter goals one by one:

```
🎯 Mass Testing Mode - Enter your goals
==================================================
Enter goals one at a time. Press Enter with empty input to finish.
Type 'quit' or 'exit' to cancel.

Goal 1: Create a simple calculator
Goal 2: Generate a random password
Goal 3: [Press Enter to finish]
```

### Advanced Options

```bash
# Enable judge evaluation for each goal
python main.py --multi_goal --judge

# Set maximum number of parallel processes
python main.py --multi_goal --max-workers 4

# Use a specific workflow template for all goals
python main.py --multi_goal --load_template <UUID>

# Combine multiple options
python main.py --multi_goal --judge --max-workers 2 --load_template <UUID>
```

## Results

Results are automatically saved in the `parallel_testing_results/` directory:

### Individual Results
- `result_process_001.json` - Results for process 1
- `result_process_002.json` - Results for process 2
- etc.

### Summary Report
- `parallel_testing_summary_<timestamp>.json` - Overall statistics and summary

### Result Structure

Each individual result contains:
```json
{
  "process_id": 1,
  "goal": "Create a simple calculator",
  "template_uuid": null,
  "judge": false,
  "human_validation": false,
  "start_time": 1642781234.567,
  "status": "completed",
  "error": null,
  "execution_time": 45.23,
  "final_uuid": "abc123...",
  "total_cost": 0.15,
  "total_rewards": 8.5
}
```

### Summary Statistics

The summary report includes:
- Total number of goals tested
- Success/failure rates
- Average execution time
- Total costs and rewards
- Detailed results for each process

## Performance Considerations

- **CPU Usage**: Each parallel process will use one CPU core
- **Memory**: Each DGM instance requires its own memory allocation
- **API Limits**: Be aware of API rate limits when running many parallel processes
- **Optimal Workers**: Generally, set max-workers to the number of CPU cores or less

## Monitoring

During execution, you'll see real-time updates:

```
🔥 Starting massive testing with 3 goals using 3 parallel processes
📊 Results will be saved to: parallel_testing_results

🚀 Starting DGM process 1 for goal: Create a simple calculator...
🚀 Starting DGM process 2 for goal: Generate a random password...
🚀 Starting DGM process 3 for goal: Create a function to validate email...

✅ Process 1 completed successfully in 42.15s
✅ Process 2 completed successfully in 38.92s
❌ Process 3 failed: API rate limit exceeded

🎉 Mass testing completed in 45.67s
📈 Summary: 2/3 goals completed successfully
📄 Summary saved to: parallel_testing_results/parallel_testing_summary_1642781234.json
```