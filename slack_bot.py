import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from engine import llm
load_dotenv()
app=App(token=os.environ['SLACK_BOT_TOKEN'])
@app.event('app_mention')
def handle_mention(event,say):
    question=event.get('text','').split('>',1)[-1].strip()
    if not question:return say('Please ask a policy question.',thread_ts=event['ts'])
    try:
        result=llm(question,indexed=True)
        say(f"*Policy assistant*\n{result['answer']}",thread_ts=event['ts'])
    except Exception as e:say(f'Could not process the question: {e}',thread_ts=event['ts'])
if __name__=='__main__':SocketModeHandler(app,os.environ['SLACK_APP_TOKEN']).start()
