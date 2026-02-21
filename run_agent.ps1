$env:PYTHONIOENCODING = "utf-8"
$prompt = Get-Content "AGENT_PROMPT.md" -Raw -Encoding UTF8
claude --dangerously-skip-permissions -p $prompt
