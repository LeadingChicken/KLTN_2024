import requests

def get_wikipedia_summary(title):
    url = "https://vi.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title,
        "explaintext": True
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Lấy nội dung từ kết quả trả về
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_content in pages.items():
        if "extract" in page_content:
            return page_content["extract"]
    return "Không tìm thấy thông tin."

# # Gọi hàm
# while True:
#     entity = input();
#     summary = get_wikipedia_summary(entity)
#     print(summary)
#     print("------------*******************------------")
