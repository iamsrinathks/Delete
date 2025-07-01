curl --request POST \
  --url 'https://YOUR_JIRA_URL/rest/api/3/search' \
  --user 'YOUR_JIRA_EMAIL:YOUR_JIRA_API_TOKEN' \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --data '{
    "jql": "project = DDBS AND issuetype = Epic AND team is not EMPTY",
    "maxResults": 5,
    "fields": [
      "summary"
    ]
  }'
