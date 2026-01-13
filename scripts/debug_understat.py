import requests
import re
import json

url = "https://understat.com/league/EPL/2024"
print(f"Testing Understat: {url}")
response = requests.get(url)
print(f"Status: {response.status_code}")

content = response.text
teams_search = re.search(r"teamsData\s+=\s+JSON\.parse\('([^']+)'\)", content)

if teams_search:
    print("Found teamsData script tag!")
    encoded_data = teams_search.group(1)
    print(f"Encoded sample: {encoded_data[:50]}...")
    decoded_data = bytes(encoded_data, 'utf-8').decode('unicode_escape')
    teams_json = json.loads(decoded_data)
    print(f"Success! Found {len(teams_json)} teams.")
else:
    print("Failed to find teamsData.")
    # Check if 'datesData' is there instead
    dates_search = re.search(r"datesData\s+=\s+JSON\.parse\('([^']+)'\)", content)
    if dates_search:
        print("Found datesData instead of teamsData.")
    else:
        print("Neither teamsData nor datesData found.")
        print("Script tags present in HTML:")
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", content, re.DOTALL)
        for i, s in enumerate(scripts):
            if "JSON.parse" in s:
                print(f"Script {i} contains JSON.parse: {s[:100]}...")
