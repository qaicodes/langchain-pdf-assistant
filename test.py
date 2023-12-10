from langchain import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-xl", task="text-generation", model_kwargs={"temperature":0, "max_length":64})