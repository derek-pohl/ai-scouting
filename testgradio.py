from gradio_client import Client, handle_file

client = Client("https://scouting.notepadded.com/")

# Submit the job (non-blocking) so we can track progress
job = client.submit(
	blue_video_path=handle_file('D:\\Mosim Builder\\MoSimBuilder-Stable - Copy\\Recordings\\BLUE_006.mp4'),
	red_video_path=handle_file('D:\\Mosim Builder\\MoSimBuilder-Stable - Copy\\Recordings\\RED_006.mp4'),
	target_fps=8,
	start_seconds=0,
	end_seconds=5,
	blue_robot_1="334",
	blue_robot_2="1919",
	blue_robot_3="2828",
	red_robot_1="5555",
	red_robot_2="3737",
	red_robot_3="4646",
	enable_robot_detection=True,
	enable_fuel_detection=True,
	api_name="/process_dual_videos"
)

# Monitor progress while job is running
print("Job submitted! Monitoring progress...")
print("-" * 50)

while not job.done():
    # Check queue status
    status = job.status()
    print(f"Status: {status}")
    
    import time
    time.sleep(2)

print("-" * 50)
print("Job complete!")
print("\n=== Result ===")
result = job.result()
print(result)
