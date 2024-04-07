from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
import os
import PyPDF2
import pytesseract
from elevenlabs import play
from elevenlabs.client import ElevenLabs

load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(
  api_key="", # Defaults to ELEVEN_API_KEY
)
gemini = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0.5, google_api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    return text

def start_working(research_paper_text):
    reader = Agent(
        role='Research Summarizer',
        goal='Provide a concise summary of the key points and insights from the research paper',
        backstory="""You are an experienced research analyst, known for your ability to quickly identify and articulate the core findings and implications of complex research papers.""",
        verbose=True,
        allow_delegation=True,
        llm=gemini
    )

    examples_giver = Agent(
        role='Concept Explainer',
        goal='Explain the key concepts from the research paper in a clear and accessible manner',
        backstory="""You are an expert educator with a talent for breaking down complex ideas and presenting them in a way that is easy for students to understand. Your explanations are grounded in real-world examples and analogies.""",
        verbose=True,
        allow_delegation=True,
        llm=gemini
    )

    question_answerer = Agent(
        role='Question Answering Agent',
        goal='Engage in a dialogue to answer questions and provide additional clarification about the research paper',
        backstory="""You are a knowledgeable subject matter expert, skilled at communicating complex information in a clear and engaging manner. You are adept at anticipating and addressing common questions or areas of confusion.""",
        verbose=True,
        allow_delegation=True,
        llm=gemini
    )

    task1 = Task(
        description=f"""Provide a concise summary of the key points, insights, and important details from the following research paper: \n{research_paper_text}\n\nYour summary should highlight the main objectives, methodology, findings, and implications of the research.""",
        agent=reader,
        expected_output="A concise summary of the key points, insights, and important details from the research paper, highlighting the main objectives, methodology, findings, and implications."
    )

    task2 = Task(
        description="""Using the summary provided, explain the core concepts from the research paper in a clear and accessible manner. Provide relatable examples and analogies to help the audience better understand the ideas.""",
        agent=examples_giver,
        expected_output="Clear and accessible explanations of the core concepts from the research paper, with relatable examples and analogies to help the audience understand the ideas."
    )

    task3 = Task(
        description="""Engage in a dialogue to answer questions and provide additional clarification about the research paper. Anticipate common questions or areas of confusion and address them in a clear and informative manner.""",
        agent=question_answerer,
        expected_output="A clear and informative dialogue that answers questions and provides additional clarification about the research paper, addressing common questions or areas of confusion."
    )

    crew = Crew(
        agents=[reader, examples_giver, question_answerer],
        tasks=[task1, task2, task3],
        verbose=2,
    )

    result = crew.kickoff()

    return result
from elevenlabs import  save

def text_to_list(text):
    text = remove_empty_lines(text)
    dialogues = []
    lines = text.strip().split("\n")
    for line in lines:
        parts = line.split(": ", 1)
        if len(parts) == 2:
            character, dialogue = parts
            dialogue = dialogue.strip().strip('"')
            dialogues.append((character, dialogue))
        else:
            # Handle the case when the line doesn't have the expected format
            print(f"Skipping line: {line}")
    return dialogues

def remove_empty_lines(text):
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip() != ""]
    result = "\n".join(non_empty_lines)
    return result



def text_to_audio(dialogue):
    dialogue = text_to_list(dialogue)

    audio_files = []
    for idx, (character, text) in enumerate(dialogue):
        print(f"Generating audio for {character}: {text}")
        if character == 'Research Summarizer':
            try:
                audio = client.generate(
                    text=text,
                    voice="Dan Dan",
                    model="eleven_multilingual_v2",
                    api_key=ELEVENLABS_API_KEY
                )
                file_path = f"research_summarizer_{idx}.wav"
                save(audio, file_path)
                audio_files.append(file_path)
            except Exception as e:
                print(f"Error generating audio for Research Summarizer: {e}")
        elif character == 'Concept Explainer':
            try:
                audio = client.generate(
                    text=text,
                    voice="Ava - conversational",
                    model="eleven_multilingual_v2",
                    api_key=ELEVENLABS_API_KEY
                )
                file_path = f"concept_explainer_{idx}.wav"
                save(audio, file_path)
                audio_files.append(file_path)
            except Exception as e:
                print(f"Error generating audio for Concept Explainer: {e}")
        elif character == 'Question Answering Agent':
            try:
                audio = client.generate(
                    text=text,
                    voice="David - professional",
                    model="eleven_multilingual_v2",
                    api_key=ELEVENLABS_API_KEY
                )
                file_path = f"question_answering_agent_{idx}.wav"
                save(audio, file_path)
                audio_files.append(file_path)
            except Exception as e:
                print(f"Error generating audio for Question Answering Agent: {e}")
        else:
            print(f"Skipping unknown character: {character}")

    print("Audio files saved.")
    return audio_files