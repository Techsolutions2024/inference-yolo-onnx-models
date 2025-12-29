from openai import OpenAI

# Cấu hình kết nối tới Antigravity
client = OpenAI(
    base_url="http://localhost:8045/v1",
    api_key="sk-9e072788add44091be9fdf2ae620e419" # Thay bằng Key thực tế trên máy bạn
)

def ask_ai_to_code(prompt):
    print("AI đang suy nghĩ và viết code...\n")
    response = client.chat.completions.create(
        model="claude-sonnet-4-5-thinking", 
        messages=[
            {"role": "system", "content": "Bạn là trợ lý lập trình chuyên nghiệp. Chỉ trả về code và giải thích ngắn gọn."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Ví dụ yêu cầu viết code
my_prompt = "Viết một script Python để tự động gửi email thông báo hàng ngày."
result = ask_ai_to_code(my_prompt)

print(result)