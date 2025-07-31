import os
import json
import time
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from openai import AzureOpenAI

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from utils import format_prompt

# ----- Synthetic dataset creation -----
def get_seed_business_ideas() -> list[str]:
    """
    Returns a list of business ideas for the synthetic dataset.
    The list was generated using AI.
    """
    return [
        # Simple
        "An online store selling handmade pottery.",
        "A local dog walking service.",
        "A blog about vegan recipes.",
        "A mobile car wash service in San Francisco.",

        # Moderate
        "A subscription box for Japanese snacks and candies.",
        "A SaaS platform for small businesses to manage their social media schedules.",
        "An AI-powered mobile app for identifying plants from a photo.",
        "A drop-shipping business focused on eco-friendly home goods.",

        # Complex
        "A B2B fintech platform that provides real-time analytics for cryptocurrency trading.",
        "A healthcare tech company using VR for physical therapy rehabilitation programs.",
        "A decentralized application (dApp) for tracking carbon credits on the blockchain.",
        "A consultancy that helps e-commerce brands optimize their supply chain using machine learning.",

        # Technology/SaaS
        "A productivity app that gamifies daily tasks and goals.",
        "A B2B platform for managing corporate travel expenses and bookings.",
        "A no-code website builder specifically for artists and creators.",
        "An AI-powered tool for generating marketing copy and ad creatives.",
        "A cloud-based tool for collaborative video editing.",
        "A platform that helps real estate agents manage their leads and listings.",
        "A social media analytics tool focused on micro-influencers.",
        "A cybersecurity service that provides penetration testing for small businesses.",
        "A software-as-a-service (SaaS) for managing employee onboarding and training.",
        "An app for tracking personal finance and setting budgets.",
        "A decentralized storage solution for digital files.",
        "A platform for creating and selling online courses.",
        "An automated customer support chatbot for e-commerce websites.",
        "A mobile app that uses augmented reality (AR) to visualize furniture in your home.",
        "A platform for connecting freelancers with short-term project opportunities.",
        "A tool for converting spoken audio into written text.",
        "A subscription service for developer tools and APIs.",
        "A software for managing non-profit donations and volunteer schedules.",
        "An online community platform for developers.",
        "A tool that simplifies the creation of smart contracts for blockchain applications.",
        "An AI assistant for drafting professional emails.",
        "A cloud-based platform for managing construction projects.",
        "A service that monitors website performance and provides optimization suggestions.",
        "A scheduling tool for medical appointments.",
        "An e-learning platform focused on teaching coding to children.",
        "A tool that automatically generates reports from raw data.",

        # E-commerce
        "A subscription box for sustainable cleaning products.",
        "An online store selling vintage clothing and accessories.",
        "A marketplace for refurbished electronics.",
        "A drop-shipping business focused on unique coffee mugs and accessories.",
        "An e-commerce store for custom-printed t-shirts and merchandise.",
        "A subscription service for craft beer.",
        "An online shop selling handmade jewelry from local artisans.",
        "A store specializing in ergonomic office furniture.",
        "An e-commerce site for gourmet spices and seasonings.",
        "A subscription box for new parents, containing baby essentials.",
        "A business selling personalized pet supplies and treats.",
        "An online store for limited-edition sneakers.",
        "A dropshipping business of custom phone cases.",
        "A marketplace for buying and selling used musical instruments.",
        "An e-commerce store focused on zero-waste products.",
        "A subscription box for baking kits with all the necessary ingredients.",
        "An online store selling niche board games and tabletop RPGs.",
        "A business selling custom digital art prints.",
        "An e-commerce platform for eco-friendly home decor.",
        "A subscription service for high-quality coffee beans from around the world.",

        # Local Services
        "A mobile detailing service for high-end cars.",
        "A personal chef service for busy professionals.",
        "A co-working space with a focus on community and networking.",
        "A mobile bike repair service.",
        "A boutique fitness studio offering high-intensity interval training (HIIT) classes.",
        "A consulting business for small restaurants looking to optimize their menus and operations.",
        "A local bakery specializing in gluten-free and vegan pastries.",
        "A professional home organizing service.",
        "A specialty coffee shop with an in-house roastery.",
        "A landscaping company focused on sustainable and drought-tolerant gardens.",
        "A cleaning service for commercial offices.",
        "A personal stylist service for creating capsule wardrobes.",
        "A business offering corporate wellness workshops and yoga sessions.",
        "A local service that helps people move and unpack.",
        "A childcare service with an educational focus.",
        "A pop-up restaurant featuring a rotating menu and guest chefs.",
        "A mobile pet grooming service.",
        "A local errand running and concierge service.",
        "A fitness studio specializing in senior citizen workouts.",

        # Creative/Media
        "A podcast production company that helps businesses launch and manage their podcasts.",
        "A freelance photography business focused on corporate headshots and branding.",
        "A digital marketing agency specializing in social media advertising for local businesses.",
        "A video production company for creating promotional videos and advertisements.",
        "A blog and newsletter focused on emerging technologies and startups.",
        "A graphic design studio for small business branding.",
        "A platform for selling stock photos and videos.",
        "A YouTube channel that reviews new tech gadgets.",
        "A creative agency that produces augmented reality (AR) experiences for brands.",
        "A freelance writing service for technical documentation.",
        "A production company for creating documentaries about social issues.",
        "A marketing agency focused on content creation for luxury brands.",
        "A business that offers drone photography and videography services.",
        "A publishing company for independent authors.",
        "A podcast about true crime stories in a specific local area.",
        "A digital agency that builds custom websites for non-profit organizations.",

        # Health & Wellness
        "A meal prep service focused on ketogenic diets.",
        "A mindfulness and meditation app with guided sessions.",
        "A brand of organic and cruelty-free skincare products.",
        "A wellness coaching service for stress management.",
        "A subscription box for natural and healthy snacks.",
        "A fitness app that provides personalized workout plans.",
        "A company that produces all-natural herbal supplements.",
        "A consulting business for creating corporate wellness programs.",
        "A service that offers in-home massage therapy.",
        "A blog about holistic health and alternative medicine.",
        "A business selling sustainable yoga mats and accessories.",
        "A mobile app for tracking and improving sleep quality.",
        "A health coaching business specializing in nutrition for athletes.",
        "A service that provides on-demand mental health support.",
        "A physical therapy clinic that specializes in sports injuries.",

        # Niche Hobbies
        "An online store selling components for custom mechanical keyboards.",
        "A subscription box for model train enthusiasts.",
        "A cafe that has a large library of board games for customers to play.",
        "A business that sells artisanal cheesemaking kits for home use.",
        "A workshop that teaches people how to build and repair their own guitars.",
        "An online community for stamp collectors.",
        "A store specializing in rare comic books and graphic novels.",
        "A subscription service for high-end fountain pens and inks.",
        "A company that offers guided photography tours.",
        "An e-commerce store for miniature painting supplies and figures.",
        "A service that restores and digitizes old family photos and videos.",
        "A business that organizes local Dungeons & Dragons campaigns.",
        "A shop specializing in vintage vinyl records.",
        "A subscription box for different kinds of teas and tea accessories.",
        "A business that creates custom-made costumes for cosplayers.",
        "A workshop that teaches woodworking and carpentry skills.",
        "A brand that sells high-quality hiking and camping gear.",
    ]
    

def get_sys_prompt() -> str:
    """
    Returns the system prompt template for the teacher LLM.
    This template was generated using AI.
    It follows the standards in prompt engineering.
    """
    
    return """
        You are an expert branding and domain name generator. Your task is to generate creative, relevant, and brandable domain name suggestions for a given business description.

        For each business description you receive, you MUST:
        1.  Generate a list of 5-7 domain name suggestions.
        2.  Ensure variety in the suggestions:
            - Include different Top-Level Domains (TLDs) like .com, .io, .ai, .co, .app, .org, .net.
            - Mix literal names (e.g., organiccoffeeco.com) with creative/abstract names (e.g., aetherbrews.com).
            - Ensure names are catchy, memorable, and easy to spell.
        3.  Format the final output as a single, valid JSON object, and nothing else. Do not include any explanatory text before or after the JSON.

        The JSON object must have two keys:
        - "business_description": The original business description I provided.
        - "domain_suggestions": A JSON list of the string domain names you generated.

        Example User Prompt:
        An organic coffee shop in a downtown area.

        Example JSON Output:
        {
            "business_description": "An organic coffee shop in a downtown area.",
            "domain_suggestions": [
                "downtownorganics.com",
                "urbanbean.co",
                "citygrind.coffee",
                "metrobrew.net",
                "organicperk.io"
            ]
        }
    """


def generate_sample(
    client : AzureOpenAI,
    business_idea : str
    ) -> str:
    """
    Generates one training data point using the teacher LLM.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": get_sys_prompt()},
                {"role": "user", "content": business_idea}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    
    except Exception as e:
        print(f"Error generating data for '{business_idea}': {e}")
        return None


def create_dataset(
    client : AzureOpenAI,
    output_path : str,
    debug : bool = False
    ) -> None:
    """
    Main function to generate the full dataset and save it as a JSON lines file.
    """

    if debug:
        print("[DEBUG] Starting synthetic dataset creation...")
    
    business_ideas = get_seed_business_ideas()
    generated_data = []

    with open(output_path, 'w') as f:
        for idea in tqdm(business_ideas, desc="Generating data samples"):
            datapoint = generate_sample(client, idea)

            if datapoint:
                f.write(json.dumps(datapoint) + '\n')
            
            time.sleep(1) # on empêche de surcharger l'api

    if debug:
        print(f"[DEBUG] Dataset creation complete.\nSaved to {output_path}")
# ----- Synthetic dataset creation -----

# Dataset class for training
class DomainDataset(Dataset):
    def __init__(self, data_files="data/training_dataset.jsonl"):
        self.raw_dataset = load_dataset("json", data_files=data_files, split="train")
        self.formatted_dataset = self.raw_dataset.map(format_prompt)
        

    def __len__(self):
        return len(self.formatted_dataset)

    def __getitem__(self, idx):
        return self.formatted_dataset.iloc[idx, 1]


if __name__ == '__main__':
    client = AzureOpenAI(
        azure_endpoint=os.environ.get("OPENAI_ENDPOINT"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_version=os.environ.get("OPENAI_API_VERSION")
    )
    
    create_dataset(
        client=client,
        output_path="data/training_dataset.jsonl",
        debug=True,
    )