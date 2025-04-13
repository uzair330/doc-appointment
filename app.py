# import os
# from dotenv import load_dotenv
# from typing import cast
# import chainlit as cl
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
# from agents.run import RunConfig

# # Load the environment variables from the .env file
# load_dotenv()

# gemini_api_key = os.getenv("GEMINI_API_KEY")

# # Check if the API key is present; if not, raise an error
# if not gemini_api_key:
#     raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


# @cl.on_chat_start
# async def start():
#     #Reference: https://ai.google.dev/gemini-api/docs/openai
#     external_client = AsyncOpenAI(
#         api_key=gemini_api_key,
#         base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     )

#     model = OpenAIChatCompletionsModel(
#         model="gemini-2.0-flash",
#         openai_client=external_client
#     )

#     config = RunConfig(
#         model=model,
#         model_provider=external_client,
#         tracing_disabled=True
#     )
#     """Set up the chat session when a user connects."""
#     # Initialize an empty chat history in the session.
#     cl.user_session.set("chat_history", [])

#     cl.user_session.set("config", config)
#     agent: Agent = Agent(name="Math Tutor", instructions="You provide help with math problems. Explain your reasoning at each step and include examples", model=model)
#     cl.user_session.set("agent", agent)

#     await cl.Message(content="Welcome to the Math Tutor AI Assistant! How can I help you today?").send()

# @cl.on_message
# async def main(message: cl.Message):
#     """Process incoming messages and generate responses."""
#     # Send a thinking message
#     msg = cl.Message(content="Loading...😊")
#     await msg.send()

#     agent: Agent = cast(Agent, cl.user_session.get("agent"))
#     config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

#     # Retrieve the chat history from the session.
#     history = cl.user_session.get("chat_history") or []
    
#     # Append the user's message to the history.
#     history.append({"role": "user", "content": message.content})
    

#     try:
#         print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
#         result = Runner.run_sync(starting_agent = agent,
#                     input=history,
#                     run_config=config)
        
#         response_content = result.final_output
        
#         # Update the thinking message with the actual response
#         msg.content = response_content
#         await msg.update()
    
#         # Update the session with the new history.
#         cl.user_session.set("chat_history", result.to_input_list())
        
#         # Optional: Log the interaction
#         print(f"User: {message.content}")
#         print(f"Assistant: {response_content}")
        
#     except Exception as e:
#         msg.content = f"Error: {str(e)}"
#         await msg.update()
#         print(f"Error: {str(e)}")

#=================================

# import os
# from dotenv import load_dotenv
# from typing import cast
# import chainlit as cl
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
# from agents.run import RunConfig

# # Load environment variables
# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# if not gemini_api_key:
#     raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# # Define specialist agents
# history_tutor_agent = Agent(
#     name="History Tutor",
#     handoff_description="Specialist agent for historical questions",
#     instructions="You provide assistance with historical queries. Explain important events and context clearly.",
#     model=None  # Will be set in start()
# )

# math_tutor_agent = Agent(
#     name="Math Tutor",
#     handoff_description="Specialist agent for math questions",
#     instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
#     model=None  # Will be set in start()
# )

# # Define triage agent
# triage_agent = Agent(
#     name="Triage Agent",
#     instructions="You analyze the user's question and decide whether to hand off to the History Tutor for history-related questions, the Math Tutor for math-related questions, or handle general queries yourself as a helpful assistant.",
#     handoffs=[history_tutor_agent, math_tutor_agent],
#     model=None  # Will be set in start()
# )

# @cl.on_chat_start
# async def start():
#     # Set up the Gemini API client
#     external_client = AsyncOpenAI(
#         api_key=gemini_api_key,
#         base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     )

#     # Define the model
#     model = OpenAIChatCompletionsModel(
#         model="gemini-2.0-flash",
#         openai_client=external_client
#     )

#     # Assign the model to all agents
#     triage_agent.model = model
#     history_tutor_agent.model = model
#     math_tutor_agent.model = model

#     # Configure the run settings
#     config = RunConfig(
#         model=model,
#         model_provider=external_client,
#         tracing_disabled=True
#     )

#     # Initialize session data
#     cl.user_session.set("chat_history", [])
#     cl.user_session.set("config", config)
#     cl.user_session.set("agent", triage_agent)  # Start with triage agent

#     await cl.Message(content="Welcome to the Panaversity AI Assistant! Ask me anything about history, math, or any topic, and I'll route you to the right tutor!").send()

# @cl.on_message
# async def main(message: cl.Message):
#     # Send a thinking message
#     msg = cl.Message(content="Thinking...")
#     await msg.send()

#     # Retrieve session data
#     agent: Agent = cast(Agent, cl.user_session.get("agent"))
#     config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
#     history = cl.user_session.get("chat_history") or []

#     # Append user's message to history
#     history.append({"role": "user", "content": message.content})

#     try:
#         print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
#         # Run the triage agent to decide who handles the query
#         result = Runner.run_sync(
#             starting_agent=agent,
#             input=history,
#             run_config=config
#         )

#         response_content = result.final_output

#         # Update the thinking message with the response
#         msg.content = response_content
#         await msg.update()

#         # Update chat history with the full conversation
#         cl.user_session.set("chat_history", result.to_input_list())

#         # Log the interaction
#         print(f"User: {message.content}")
#         print(f"Assistant: {response_content}")

#     except Exception as e:
#         msg.content = f"Error: {str(e)}"
#         await msg.update()
#         print(f"Error: {str(e)}")

import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

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
    # Send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()

    # Retrieve session data
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    # Get or initialize chat history
    history = cl.user_session.get("chat_history") or []

    # Append user's message to history
    history.append({"role": "user", "content": message.content})

    try:
        # Log the incoming message, history, and initial agent
        print(f"\n[USER_MESSAGE] {message.content}")
        print(f"[CHAT_HISTORY] {history}")
        print(f"[CALLING_AGENT] Starting with {agent.name}")

        # Run the triage agent with the full chat history
        result = Runner.run_sync(
            starting_agent=agent,
            input=history,
            run_config=config
        )

        response_content = result.final_output

        # Infer the responding agent for logging
        if "brother" in message.content.lower():
            inferred_agent = "Brother Agent"
        elif "daughter" in message.content.lower():
            inferred_agent = "Daughter Agent"
        else:
            inferred_agent = "Triage Agent (general response)"
        
        print(f"[AGENT_RESPONSE] Handled by {inferred_agent}")
        print(f"[RESPONSE_CONTENT] {response_content}")

        # Update the thinking message with the response
        msg.content = response_content
        await msg.update()

        # Update chat history with the latest conversation
        cl.user_session.set("chat_history", result.to_input_list())

        # Log the updated history
        print(f"[CHAT_HISTORY_UPDATED] {result.to_input_list()}")

    except Exception as e:
        # Log the error
        print(f"[ERROR] Exception occurred: {str(e)}")
        msg.content = f"Error: {str(e)}"
        await msg.update()