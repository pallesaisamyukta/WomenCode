import streamlit as st
import openai
from scripts.retrieval import PromptGenerator
import re

# Title of the app
st.title("WomenCode")
# Display the welcome message
st.markdown("""
**Welcome!**

🩸 **Menstrual Cycle Queries:**  
Feel free to ask questions about menstrual cycles, phases, diet options, or lifestyle changes required.

🍽️ **Recipe Generation:**  
To generate a recipe, please start your query with:  
"generate a recipe:- "  
You can also specify additional criteria:  
- **Tags:** Include tags like vegan, vegetarian, etc.  
- **Nutrition:** Specify nutrition preferences like low-carb, high-protein, etc.  
- **Ingredients:** List some available ingredients at home, e.g., onion, etc.  
- **Rating:** If you want only top-rated recipes, mention it too.
Example - generate a recipe:- Tags: ['60-minutes-or-less', 'time-to-make', 'course', 'preparation', 'pancakes-and-waffles', 'breakfast', 'easy', 'kid-friendly', 'dietary', 'low-sodium', 'low-in-something'] Nutrition: low-calories, low-total fat, high-sugar, low-sodium, low-protein, low-carbohydrates Ingredients:  'eggs','nonstick cooking spray', 'evaporated milk', 'strawberries', 'nutmeg', 'shredded wheat cereal', 'italian bread' Rating: 5.0
Ask away, and I'll do my best to assist you!
""")

# Paths and configurations
json_path = './data/processed/chunks.json'
openai.api_key = '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'

prompt_generator = PromptGenerator(json_path)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

store_time = []
i = 0
if user_input := st.chat_input("What is up?"):

    if len(user_input) < 10:
        st.write("Please re-write the full question.")
    
    # If the question is on a recipe then going to BART Finetuned model
    elif re.search(r'\b(generate|suggest) a recipe\b', user_input, re.IGNORECASE):
        st.session_state.messages.append({"role": "user", "content": user_input})
        user_input = user_input.split(":-")[1]
        prompt = prompt_generator.recipe_details_prompt(user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = [
                        {"role": "system", "content": prompt}
                    ],

                    max_tokens=4096,
                )
                # Display the generated text
            st.write(response['choices'][0]['message']['content'])
        st.session_state.messages.append({"role": "assistant", "content":response['choices'][0]['message']['content']})

    else:
        # Considering the last user question to append to the present question if required.
        if len(st.session_state.messages) > 2:
            # If the last question was to generate a recipe then not appending.
            x = st.session_state.messages[-2]['content'].str.contains('generate a recipe')
            print(x)
            if ~x:
                i = -2
        
        prev_user_q = " ".join([m["content"] for m in st.session_state.messages[i:] if m["role"] == "user"])

        # Generate prompt
        prompt = prompt_generator.rag_prompt(user_input, prev_user_q)

        # Adding to the prompt, so that generated response is not repeated.
        prev_ans = " ".join([m["content"] for m in st.session_state.messages[i:] if m["role"] == "assistant"])
        
        if prev_ans:
            prompt = prompt + "Consider the previous messages in the chat and don't repeat it. Previous responses were - " + str(prev_ans)
            print(prompt)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Calling ChatGPT API for generating the response.
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = [
                    {"role": "system", "content": prompt}
                ],

                max_tokens=4096,
            )
            # Display the generated text
            st.write(response['choices'][0]['message']['content'])
        # Saving all the messages.
        st.session_state.messages.append({"role": "assistant", "content":response['choices'][0]['message']['content']})

        print("I'm done")
