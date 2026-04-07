import os
import re
from datetime import datetime
from math_verify import parse, verify # TODO:验证数学公式是否等价

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    solution = [a['content'] for a in assistant]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                ground_truth = sol.strip()
                student_answer = content.strip()
                
                ground_truth = ground_truth.removesuffix('.')
                student_answer = student_answer.removesuffix('.')

                # Compare the extracted answers
                if student_answer.lower() == ground_truth.lower():
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)

        log_path = '/data/dongxinpeng/my_former_work/work/log.txt'
        # current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"------------- Accuracy reward: {reward} -------------\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")
    return rewards


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]
