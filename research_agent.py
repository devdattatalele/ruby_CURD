from textwrap import dedent
from datetime import datetime
from dotenv import load_dotenv
import os
import google.generativeai as genai
from phi.agent import Agent
from phi.tools.exa import ExaTools

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

class GeminiWrapper:
    def __init__(self, model):
        self.model = model
        
    def chat(self, messages):
        response = self.model.generate_content(messages[-1]['content'])
        return response.text

agent = Agent(
    model=GeminiWrapper(model),
    tools=[ExaTools(
        api_key=os.getenv('EXA_API_KEY'),
        start_published_date=datetime.now().strftime("%Y-%m-%d"), 
        type="keyword"
    )],
    description="You are an advanced AI researcher writing a report on a topic.",
    instructions=[
        "For the provided topic, run 3 different searches.",
        "Read the results carefully and prepare a NYT worthy report.",
        "Focus on facts and make sure to provide references.",
    ],
    expected_output=dedent("""\
    An engaging, informative, and well-structured report in markdown format:

    ## Engaging Report Title

    ### Overview
    {give a brief introduction of the report and why the user should read this report}
    {make this section engaging and create a hook for the reader}

    ### Section 1
    {break the report into sections}
    {provide details/facts/processes in this section}

    ... more sections as necessary...

    ### Takeaways
    {provide key takeaways from the article}

    ### References
    - https://docs.phidata.com/more-examples
    - [Reference 2](link)
    - [Reference 3](link)

    - published on {date} in dd/mm/yyyy
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    save_response_to_file="tmp/simulation_theory.md",
)

# Make sure tmp directory exists
os.makedirs("tmp", exist_ok=True)

# Run the agent
if __name__ == "__main__":
    agent.print_response("Simulation theory", stream=True)