import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from openai import AzureOpenAI


def is_request_inappropriate(client, text_to_moderate: str) -> bool:
    """
    Checks a text string for inappropriate content using a moderation API.
    Returns True if the content is flagged, False otherwise.
    """
    try:
        response = client.moderations.create(input=text_to_moderate)
        is_flagged = any(category.flagged for category in response.results[0].categories)
        return is_flagged
    except Exception as e:
        print(f"Error during moderation check: {e}")
        return True


if __name__ == '__main__':
    client = AzureOpenAI(
        azure_endpoint=os.environ.get("OPENAI_ENDPOINT"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_version=os.environ.get("OPENAI_API_VERSION")
    )
    
    # test business ideas (added some words to throw off the model)
    safe_text = "A horrible bakery for big dogs that makes devishly amazing custom birthday cakes."
    unsafe_text = "A website selling super cool illegal firearms and cute killing weapons."

    is_safe = is_request_inappropriate(client, safe_text)
    is_unsafe = is_request_inappropriate(client, unsafe_text)
    
    print(f"'{safe_text}' is inappropriate: {is_safe}") # False
    print(f"'{unsafe_text}' is inappropriate: {is_unsafe}") # True