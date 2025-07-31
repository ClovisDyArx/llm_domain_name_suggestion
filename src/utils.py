


def format_prompt(sample : dict):
    """
    Formats a single data sample into the Gemma-2 chat template.
        sample : {"business_description" : 'blablabla..', "domain_suggestions" : ['blabla.com', 'bloblo.fr', ..]}
        =>
        "<start_of_turn>user\nGenerate ... <end_of_turn>"
    """
    domains_str = ", ".join(sample['domain_suggestions'])
    
    text = f"<start_of_turn>user\nGenerate a diverse list of high-quality domain names for the following business: {sample['business_description']}<end_of_turn>\n<start_of_turn>model\n{domains_str}<end_of_turn>"
    return {"text": text}