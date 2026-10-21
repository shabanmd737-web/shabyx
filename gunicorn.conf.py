# Gunicorn reads this file automatically from the project root.
# Gemini can take a while to answer, so give each request up to 120 seconds
# and let one worker handle several users at the same time.
workers = 1
worker_class = "gthread"
threads = 4
timeout = 120
