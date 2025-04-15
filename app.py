import os
from dotenv import load_dotenv
from typing import cast, Optional
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig
from openai.types.responses import ResponseTextDeltaEvent
from sqlmodel import SQLModel, Field, Session, create_engine, select
from datetime import datetime, timedelta
import secrets
import string
from email.message import EmailMessage
import smtplib

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

DATABASE_URL = os.getenv('DR_URL')
if not DATABASE_URL:
    raise ValueError("DR_URL is not set. Please ensure it is defined in your .env file.")

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
if not MAIL_USERNAME or not MAIL_PASSWORD:
    raise ValueError("MAIL_USERNAME and MAIL_PASSWORD must be set in your .env file.")

engine = create_engine(DATABASE_URL)

# Define database models
class Doctor(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    specialty: str

class Patient(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str
    unique_id: str

class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    doctor_id: int
    patient_id: int
    date: str  # YYYY-MM-DD
    time: str  # HH:MM

# Initialize database: create tables and add sample doctors if none exist
def init_database():
    """Create database tables and populate with sample doctors if empty."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        if session.exec(select(Doctor)).first() is None:
            sample_doctors = [
                Doctor(name="Dr. Smith", specialty="Cardiology"),
                Doctor(name="Dr. Johnson", specialty="Neurology"),
                Doctor(name="Dr. Williams", specialty="Pediatrics"),
            ]
            session.add_all(sample_doctors)
            session.commit()

# Helper functions
def generate_unique_id() -> str:
    """Generate a unique ID in the format 'Uza-XXXXX'."""
    characters = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(characters) for _ in range(5))
    return f"Uza-{random_part}"

def get_patient_by_name(name: str) -> Optional[Patient]:
    """Retrieve a patient from the database by name."""
    with Session(engine) as session:
        return session.exec(select(Patient).where(Patient.name == name)).first()

def get_doctor_by_name(name: str) -> Optional[Doctor]:
    """Retrieve a doctor from the database by name."""
    with Session(engine) as session:
        return session.exec(select(Doctor).where(Doctor.name == name)).first()

# Email sending function
def send_email(subject: str, body: str, to_email: str):
    """Send an email using Gmail's SMTP server."""
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = MAIL_USERNAME
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.send_message(msg)
        print(f"Email sent successfully to {to_email}")
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")

# Define custom tools
@function_tool
def list_doctors() -> str:
    """List all available doctors with their specialties."""
    with Session(engine) as session:
        doctors = session.exec(select(Doctor)).all()
        if not doctors:
            return "No doctors available."
        doctor_list = ", ".join([f"{d.name} ({d.specialty})" for d in doctors])
        return f"Available doctors: {doctor_list}"

@function_tool
def register_patient(name: str, email: str) -> str:
    """Register a new patient with name and email."""
    with Session(engine) as session:
        existing = session.exec(select(Patient).where(Patient.email == email)).first()
        if existing:
            return f"Email {email} is already registered."
        unique_id = generate_unique_id()
        patient = Patient(name=name, email=email, unique_id=unique_id)
        session.add(patient)
        session.commit()
        return f"Patient registered: {name}, Email: {email}, Unique ID: {unique_id}"

@function_tool
def check_availability(doctor_name: str, date: str) -> str:
    """Check available time slots for a doctor on a specific date."""
    doctor = get_doctor_by_name(doctor_name)
    if not doctor:
        return f"Doctor {doctor_name} not found."
    start_time = datetime.strptime("09:00", "%H:%M")
    end_time = datetime.strptime("17:00", "%H:%M")
    slots = []
    current = start_time
    while current <= end_time:
        slots.append(current.strftime("%H:%M"))
        current += timedelta(minutes=30)
    with Session(engine) as session:
        booked = session.exec(
            select(Appointment.time).where(
                Appointment.doctor_id == doctor.id,
                Appointment.date == date
            )
        ).all()
        available = [slot for slot in slots if slot not in booked]
        if available:
            return f"Available slots for {doctor_name} on {date}: {', '.join(available)}"
        return f"No available slots for {doctor_name} on {date}."

@function_tool
def book_appointment(patient_name: str, doctor_name: str, date: str, time: str) -> str:
    """Book an appointment for a patient with a doctor on a specific date and time."""
    patient = get_patient_by_name(patient_name)
    if not patient:
        return f"Patient {patient_name} not found. Please register first."
    doctor = get_doctor_by_name(doctor_name)
    if not doctor:
        return f"Doctor {doctor_name} not found."
    with Session(engine) as session:
        existing = session.exec(
            select(Appointment).where(
                Appointment.doctor_id == doctor.id,
                Appointment.date == date,
                Appointment.time == time
            )
        ).first()
        if existing:
            return "Sorry, that slot is already booked."
        appointment = Appointment(
            doctor_id=doctor.id,
            patient_id=patient.id,
            date=date,
            time=time
        )
        session.add(appointment)
        session.commit()
        # Send confirmation email
        subject = "Appointment Confirmation"
        body = f"Dear {patient.name},\n\nYour appointment with {doctor.name} on {date} at {time} is confirmed.\n\nYour unique ID: {patient.unique_id}\n\nThank you."
        send_email(subject, body, patient.email)
        return "Appointment booked successfully. Confirmation email sent."

@function_tool
def phone_book_appointment(patient_name: str, email: str, doctor_name: str, date: str, time: str) -> str:
    """Simulate a phone-based appointment booking, registering the patient if necessary."""
    patient = get_patient_by_name(patient_name)
    if not patient:
        register_msg = register_patient(patient_name, email)
        patient = get_patient_by_name(patient_name)
        if not patient:
            return f"Failed to register patient: {register_msg}"
    book_msg = book_appointment(patient_name, doctor_name, date, time)
    return f"Simulated phone booking: {book_msg}"

# Define the Appointment Agent
appointment_agent = Agent(
    name="Appointment Agent",
    instructions="You are an appointment booking assistant. Help patients by registering them, listing doctors, checking availability, booking appointments, and simulating phone bookings. Use the provided tools to perform these tasks.",
    tools=[list_doctors, register_patient, check_availability, book_appointment, phone_book_appointment],
    model=None  # Will be set in on_chat_start
)

@cl.on_chat_start
async def start():
    """Initialize the Doctor Appointment System on chat start."""
    print("[SESSION_START] Initializing Doctor Appointment System")
    init_database()  # Ensure database is ready
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )
    appointment_agent.model = model
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    cl.user_session.set("agent", appointment_agent)
    print("[AGENT_SETUP] Appointment Agent initialized with tools: list_doctors, register_patient, check_availability, book_appointment, phone_book_appointment")
    await cl.Message(content="Welcome to the Doctor Appointment System! You can register patients, list doctors, check availability, book appointments, or simulate phone bookings.").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages and process them with the Appointment Agent."""
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})
    msg = cl.Message(content="")
    await msg.send()
    try:
        print(f"\n[USER_MESSAGE] {message.content}")
        print(f"[CHAT_HISTORY] {history}")
        print(f"[CALLING_AGENT] Appointment Agent")
        result = Runner.run_streamed(
            starting_agent=agent,
            input=history,
            run_config=config
        )
        response_content = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                delta = event.data.delta
                if delta:
                    response_content += delta
                    await msg.stream_token(delta)
        print(f"[AGENT_RESPONSE] {response_content}")
        await msg.update()
        history.append({"role": "assistant", "content": response_content})
        cl.user_session.set("chat_history", history)
        print(f"[CHAT_HISTORY_UPDATED] {history}")
    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        msg.content = f"Error: {str(e)}"
        await msg.update()