from datetime import datetime
from ai_benchmark import AIBenchmark

benchmark = AIBenchmark(use_CPU=None, verbose_level=3)

start_now = datetime.now()
start_current_time = start_now.strftime("%H:%M:%S")
print("Start Time =", start_current_time)
benchmark.run()
end_now = datetime.now()
end_current_time = end_now.strftime("%H:%M:%S")
print("End Time =", end_current_time)
diff = end_now-start_now
diff_minutes = diff.total_seconds() / 60
print("Execution time in minutes =", diff_minutes)
