import csv
import io
from xml.dom import minidom
import sys

def get_text(node):
    """Extract text from node (if any)"""
    if node.firstChild and node.firstChild.nodeType == node.firstChild.TEXT_NODE:
        return node.firstChild.data
    return ""

def convert_xml_to_csv(xml_file, csv_file):
    """Convert the XML file with medical QA data to a CSV file with Question,Answer columns"""
    try:
        # Parse the XML file
        xmldoc = minidom.parse(xml_file)
        
        # Create a list to store our data
        data = []
        
        # Process each NLM-QUESTION
        for question in xmldoc.getElementsByTagName('NLM-QUESTION'):
            # Extract the original question
            original_questions = question.getElementsByTagName('Original-Question')
            if original_questions:
                original_question = original_questions[0]
                subject_elements = original_question.getElementsByTagName('SUBJECT')
                message_elements = original_question.getElementsByTagName('MESSAGE')
                
                # Combine subject and message
                q_text = ""
                if subject_elements:
                    subject_text = get_text(subject_elements[0])
                    if subject_text:
                        q_text += subject_text + ". "
                
                if message_elements:
                    message_text = get_text(message_elements[0])
                    if message_text:
                        q_text += message_text
                
                q_text = q_text.strip()
                
                # Extract all reference answers
                ref_answers_elements = question.getElementsByTagName('ReferenceAnswers')
                if ref_answers_elements:
                    ref_answers = ref_answers_elements[0]
                    all_answers = []
                    
                    for ref_answer in ref_answers.getElementsByTagName('ReferenceAnswer'):
                        answer_elements = ref_answer.getElementsByTagName('ANSWER')
                        if answer_elements:
                            answer_text = get_text(answer_elements[0])
                            if answer_text:
                                all_answers.append(answer_text.strip())
                    
                    # Combine all answers
                    combined_answer = ". ".join(all_answers)
                    
                    # Add to our data if we have both question and answer
                    if q_text and combined_answer:
                        data.append((q_text, combined_answer))
        
        # Write to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Answer"])
            writer.writerows(data)
        
        print(f"Converted {len(data)} question-answer pairs to {csv_file}")
        return True
    
    except Exception as e:
        print(f"Error converting XML to CSV: {e}")
        return False

# Execute the script
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python xml_to_csv.py input.xml output.csv")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = convert_xml_to_csv(input_file, output_file)
    if success:
        print(f"Successfully converted {input_file} to {output_file}")
    else:
        print(f"Failed to convert {input_file} to {output_file}")
