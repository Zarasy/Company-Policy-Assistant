# Company Policy Assistant

## Setup
1. Install Python 3.10+.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in keys. Do not commit `.env`.
4. Run `streamlit run app.py` (export GEMINI_API_KEY or set Streamlit secrets first).
5. Run `python slack_bot.py` to start Slack bot.

## Streamlit deployment
Push files to a GitHub repository, then create an app at https://share.streamlit.io/ with `app.py` as entrypoint. Add `GEMINI_API_KEY = "..."` under app Secrets.

## Slack setup
Create a separate workspace and channel. Create an app at https://api.slack.com/apps. Enable Socket Mode and create an app-level token with `connections:write`. Add bot OAuth scopes `app_mentions:read` and `chat:write`. Subscribe to bot event `app_mention`, install app in workspace, invite it to test channel. Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in local `.env`. Run bot, mention it in channel, capture screenshot. Invite raz@sdu.dk to the test channel (workspace guest permissions may apply).

## Important methodological limitations
The indexed approach uses a **local TF-IDF vector index**, not semantic embedding vectors. This meets a literal vector-index comparison but is less semantically capable than embedding-based RAG. The indexed method reports only LLM tokens, not TF-IDF operations. Response times vary by API and network. Unsupported-answer tendency must be established by testing representative questions and checking answers against CSV source text; do not present unmeasured rates as facts. The full-context approach can approach model context limits as data grows.
