import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from openai import AzureOpenAI


def is_request_inappropriate(
    client,
    text_to_moderate: str,
    moderation_deployment_name : str, # azure moderation model
    ) -> bool:
    """
    Checks a text string for inappropriate content using a moderation API.
    Returns True if the content is flagged, False otherwise.
    """
    try:
        response = client.moderations.create(
            model=moderation_deployment_name,
            input=text_to_moderate,
        )
        is_flagged = any(category.flagged for category in response.results[0].categories)
        return is_flagged
    except Exception as e:
        print(f"Error during moderation check: {e}")
        return True


if __name__ == '__main__':
    MODERATION_DEPLOYMENT_NAME = os.environ.get("OPENAI_MODERATION_DEPLOYMENT") 

    client = AzureOpenAI(
        azure_endpoint=os.environ.get("OPENAI_ENDPOINT"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_version=os.environ.get("OPENAI_API_VERSION")
    )
    
    test_cases = {
        "safe_text": "A bakery for dogs that makes custom birthday cakes.",
        "unsafe_text": "A website selling illegal firearms and weapons.",
        "safe_tricky_text": "A horrible bakery for big dogs that makes devishly amazing custom birthday cakes.",
        "unsafe_tricky_text": "A website selling super cool illegal firearms and cute killing weapons."
    }

    for name, text in test_cases.items():
        is_inappropriate = is_request_inappropriate(client, text, MODERATION_DEPLOYMENT_NAME)
        print(f"'{text}' is inappropriate: {is_inappropriate}")