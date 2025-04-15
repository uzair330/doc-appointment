import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from openai.types.responses import ResponseTextDeltaEvent
from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional
import os

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


# Connection to Neon Database
DATABASE_URL = os.getenv('DR_URL')
engine = create_engine(DATABASE_URL)

# Define the two specialized agents
brother_agent = Agent(
    name="Brother Agent",
    handoff_description="Responds with 'Khubaib' when user asks about brother",
    instructions="If the user's message contains 'brother', respond only with 'Khubaib'. For other queries, do not respond.",
    model=None
)

daughter_agent = Agent(
    name="Daughter Agent",
    handoff_description="Responds with 'Ateeqa' when user asks about daughter",
    instructions="If the user's message contains 'daughter', respond only with 'Ateeqa'. For other queries, do not respond.",
    model=None
)

# Define triage agent to route queries
triage_agent = Agent(
    name="Triage Agent",
    instructions="Analyze the user's message and the chat history. If the message contains 'brother', hand off to the Brother Agent. If it contains 'daughter', hand off to the Daughter Agent. For all other queries, respond as a helpful assistant, using the chat history to stay consistent.",
    handoffs=[brother_agent, daughter_agent],
    model=None
)

@cl.on_chat_start
async def start():
    # Log session start
    print("[SESSION_START] Initializing Panaversity AI Assistant")

    # Set up the Gemini API client
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Define the model
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    # Assign the model to all agents
    triage_agent.model = model
    brother_agent.model = model
    daughter_agent.model = model

    # Configure the run settings
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # Initialize session data with empty chat history
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    cl.user_session.set("agent", triage_agent)

    # Log agent setup
    print(f"[AGENT_SETUP] Triage Agent initialized with handoffs to: {', '.join([a.name for a in triage_agent.handoffs])}")

    await cl.Message(content="Welcome to the AI Assistant! Ask about your brother or daughter, or anything else!").send()

@cl.on_message
async def main(message: cl.Message):
    # Retrieve session data
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    # Get or initialize chat history
    history = cl.user_session.get("chat_history") or []

    # Append user's message to history
    history.append({"role": "user", "content": message.content})

    # Create a message for streaming
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Log the incoming message, history, and initial agent
        print(f"\n[USER_MESSAGE] {message.content}")
        print(f"[CHAT_HISTORY] {history}")
        print(f"[CALLING_AGENT] Starting with {agent.name}")

        # Run the agent with streaming
        result = Runner.run_streamed(
            starting_agent=agent,
            input=history,
            run_config=config
        )

        # Stream the response
        response_content = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                delta = event.data.delta
                if delta:  # Only stream non-empty deltas
                    response_content += delta
                    await msg.stream_token(delta)

        # Infer the responding agent for logging
        if "brother" in message.content.lower():
            inferred_agent = "Brother Agent"
        elif "daughter" in message.content.lower():
            inferred_agent = "Daughter Agent"
        else:
            inferred_agent = "Triage Agent (general response)"
        
        print(f"[AGENT_RESPONSE] Handled by {inferred_agent}")
        print(f"[RESPONSE_CONTENT] {response_content}")

        # Finalize the message
        await msg.update()

        # Update chat history with the full response
        history.append({"role": "assistant", "content": response_content})
        cl.user_session.set("chat_history", history)

        # Log the updated history
        print(f"[CHAT_HISTORY_UPDATED] {history}")

    except Exception as e:
        # Log the error
        print(f"[ERROR] Exception occurred: {str(e)}")
        msg.content = f"Error: {str(e)}"
        await msg.update()