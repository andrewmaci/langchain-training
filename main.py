import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information="""
    The Russo-Ukrainian war began in February 2014 and is ongoing. Following Ukraine's Revolution of Dignity, Russia occupied Crimea and annexed it from Ukraine. It then supported Russian separatist armed groups who started a war in the eastern Donbas region against Ukraine's military. In 2018, Ukraine declared the region to be occupied by Russia.[8] The first eight years of conflict also involved naval incidents and cyberwarfare. In February 2022, Russia launched a full-scale invasion of Ukraine and began occupying more of the country, starting the current phase of the war, the biggest conflict in Europe since World War II. The war has resulted in a refugee crisis and hundreds of thousands of deaths.
    """
    summary_template = """
    Given the {information} about the fact i want you to:
    1. Provide a short summary
    2. Tell me who is it about?
    """
    fact_summary_template = PromptTemplate(
        input_variables=['information'],
        template=summary_template
    )
    llm = ChatOllama(
        temperature=0.1,
        base_url=os.getenv("OLLAMA_HOST"),
        model="hf.co/speakleash/Bielik-11B-v3.0-Instruct-GGUF:Q4_K_M"
    )

    chain = fact_summary_template | llm
    response = chain.invoke(input={"information":information})

    print(response.content)

if __name__ == "__main__":
    main()
