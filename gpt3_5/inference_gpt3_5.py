import os
import json
import openai
from typing import List
# from openai import OpenAI

openai.api_type = "azure"
openai.api_base = "https://gpt35newdec23.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
openai.api_key = "45a56bedd7d54f30ab4a622cdce4803d"

with open("video_desc.json", 'r') as f:
    video_desc = json.load(f)

def create_prompt(video_desc: str,
                  qn_ans: str) -> str:
    instr = '''
      TASK:

        With the video summary as input, a question with multiple choice options and the correct option is provided. 
        Using the video summary, create four wrong options for the question using the given correct option (ignore the original wrong options). 
        Also, for each generated option, provide the method by which the wrong option was generated from the question and answer.

      GUIDELINES:

      - The options should be statements, NOT questions.
      - The options should NOT answer the question correctly.
      - The options should be coherent and grammatically correct.
      - The options should be created using the living things (e.g., woman, swimmer, horse), objects (e.g., hammer, bowl), places (e.g., meadow, house) and associated events (e.g., playing, eating) mentioned in the video summary as much as possible.
      - Option generation should be done keeping in mind that the video description might be noisy and there might be undetected or misclassified objects in it.
      - The options should be diverse and not repetitive in meaning.

      e.g., 

        Video Summary:
          The photographer waited until the sun was setting to capture the perfect picture. 
          [person_1] and [person_2] were all smiles as they posed for the perfect wedding picture by the beach in the backdrop of the sunset.
          Meanwhile, [person_3] and [person_4], though not in the picture, were standing nearby, giggling and talking among themselves.

        (1) 
          QUESTION: 
            Why are [person_1] and [person_2] smiling?

          QUESTION TYPE:
            Explanatory

          ANSWER:
            Because [person_1] and [person_2] are posing for their wedding photo.

          GENERATED WRONG OPTIONS:

            a. <OPT> Because [person_1] got injured. </OPT> <MET> Since [person_1] and [person_2] are smiling, it must be a happy occasion. Since the opposite of happy is sad/painful, generate a wrong option describing a sad/painful experience, i.e. injury. </MET>    
            b. <OPT> Because [person_1] and [person_2] are enjoying playing football. </OPT> <MET> Since [person_1] and [person_2] are smiling, it must be a happy occasion. So, generate a wrong option by replacing the wedding with another happy activity, i.e. enjoying playing football. </MET>
            c. <OPT> Because [person_3] and [person_4] are posing for their wedding photo. </OPT> <MET> [person_1] and [person_2] are getting married. So, generate a wrong option by replacing the people getting married with [person_3] and [person_4], who are mentioned in the video summary but not getting married. </MET>
            d. <OPT> Because [person_1] and [person_2] are enjoying the sunrise. </OPT> <MET> The video summary mentions that there is a sunset at the time of the wedding. So, generate a wrong option by keeping the wedding as the same but changing the time of day, i.e., sunrise. </MET>

        (2)

          QUESTION: 
            What would happen if [person_3] and [person_4] photobombed?

          QUESTION TYPE:
            Counterfactual

          ANSWER:
            [person_1] and [person_2] would not have the perfect wedding photo at sunset by the beach.

          GENERATED WRONG OPTIONS:

            a. <OPT> [person_1] and [person_2] would not have the perfect wedding photo at sunrise by the beach. </OPT> <MET> The video summary mentions that there is a sunset at the time of the wedding. So, generate a wrong option by keeping the wedding as the same but changing the time of day, i.e., sunrise. </MET>
            b. <OPT> [person_1] and [person_2] would be thrilled. </OPT> <MET> If someone is photobombing your wedding picture, you might be irritated or angry. So, generate a wrong option by adding the opposite emotion, i.e., thrilled. </MET>
            c. <OPT> [person_1] and [person_2] would not have the perfect wedding photo at sunset by the lake. </OPT> <MET> The video summary mentions that the location is a sea beach. So, generate a wrong option by replacing with another location, i.e. lake. </MET>
            d. <OPT> [person_1] and [person_2] would be anxious. </OPT> <MET> If someone is photobombing your wedding picture, you might be irritated or angry. So, generate a wrong option by adding an unrelated emotion, i.e., anxious. </MET>

      '''

        # - GUIDELINES: 
        #     The wrong options can be created from the correct option by replacing the key words with different words which change the sentence context.

        #   e.g., 
          
        #   Correct option:
        #     The <boy> is <drinking> a <can of soda>.
 
        #   (Possible) wrong options:
        #     -> <OPT> The <girl> is <eating> an <apple> </OPT> <EXPL> ... </EXPL>   # Replace <boy> with <girl>, <drinking> with <eating>, <can of soda> with <apple>
        #     -> <OPT> The <boy> is <playing> a <guitar>. </OPT> <EXPL> ... </EXPL>  # Replace <drinking> with <playing>, <can of soda> with <guitar>
        #     -> <OPT> The <woman> is <baking> a <cake>. </OPT> <EXPL> ... </EXPL> # Replace <boy> with <woman>, <drinking> with <baking>, <can of soda> with <cake>

        #   Additionally, if the video description mentions :
          
        #     - specific places (e.g., table),
        #     - events(e.g, party)
            
        #   which aren't mentioned in the correct option, they can also be modified and added to make wrong options.
        
        #   e.g., 

        #    Video Summary:

        #       ... The children were enjoying a can of soda sitting at the <table>. The <party> was going on till late. ...

        #    Correct option:
        #     The <boy> is <drinking> a <can of soda>.

        #    Events/places not mentioned in correct option:
        #     - <table>, <party>

        #     (Possible) wrong options:
        #     -> <OPT> The <boy> is <drinking> a <can of soda> in the <meeting>. </OPT> <EXPL> ... </EXPL> (Add <party> to option, replace with another event, i.e., <meeting>).
        #     -> <OPT> The <boy> is <drinking> a <can of soda> sitting on the <sofa>. <OPT> <EXPL> ... </EXPL> (Add <table> to option, replace with another place of sitting, i.e., <sofa>)
    
    prompt = f'''{instr}
    
              VIDEO SUMMARY: 
              
               {video_desc}
              
              {qn_ans}
            '''
    return prompt

qn_type = "counterfactual"
qn_ans = f'''
                  QUESTION: 
                    What would happen if [person_2] cut into [person_1]?

                  QUESTION TYPE:
                    {qn_type}
                    
                  ANSWER:
                    [person_1] might have to get off the boat and seek medical attention.

                  GENERATED WRONG OPTIONS:

                '''

                  # OPTIONS:
                  
                  #   a. Maybe the athletes won't play.
                  #   b. Juice would splash everywhere.
                  #   c. [person_1] would be surprised.
                    
                  #   e. [person_1] will break a chessman down.

                  # QUESTION: Why are [person_1] and [person_2] holding swords?
                  # OPTIONS:
                    
                  #   a. Because the [person_1] prepared to jump.
                  #   b. [person_1] is doing it for fun.
                  #   c. [person_2] doesn't seem to want [person_2] to touch him.
                  #   d. [person_1] and [person_2] are trying to practice hula hooping.
                  #   e. [person_1] and [person_2] are fighting swords.

                  # ANSWER: 
                  #   e. [person_1] and [person_2] are fighting swords.

                  # WRONG OPTIONS:

# qn_type = "explanatory"
prompt = create_prompt(video_desc, qn_type, qn_ans)

chat_completion = openai.Completion.create(
  engine="gpt35tdec23",
  prompt=prompt,
  temperature=1,
  max_tokens=330,
  top_p=0.5,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print(chat_completion["choices"][0]["text"])
