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

# Đọc file và xử lý từng entity
def process_entities():
    try:
        with open('entities.txt', 'r', encoding='utf-8') as file:
            for line in file:
                entity = line.strip()
                try:
                    summary = get_wikipedia_summary(entity)
                    print(f"\nEntity: {entity}")
                    print(f"Summary: {summary}\n")
                    print("----------------------------------------------")
                except Exception as e:
                    print(f"Error processing entity '{entity}': {str(e)}")
    except FileNotFoundError:
        print("Could not find entities.txt file")
    except Exception as e:
        print(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    process_entities()


