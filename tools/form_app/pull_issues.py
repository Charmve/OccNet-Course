import requests as re

# Github 认证 token
token = "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"

# 请求头
headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github.v3+json",
}

# 请求体
data = {"title": "Issue Title", "body": "Issue Body"}


def pull_issues(data=data):
    # 发送 POST 请求
    # https://github.com/Charmve/100days
    repo_owner = "Charmve"
    repo_name = "OccNet-Course"
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    response = re.post(url, headers=headers, json=data)

    # 打印结果
    if response.status_code == 201:
        issue_url = response.json()["html_url"]
        print("Issue 已经成功创建！")
        print(f"{issue_url}")
    else:
        print(f"创建 Issue 失败，错误码：{response.status_code}")


def close_github_issues(issue_url):
    pattern = r"issues\/(\d+)"
    match = re.search(pattern, issue_url)
    if match:
        issue_number = match.group(1)
        print(issue_number)
    else:
        print("No issue number found in the URL.")

    # 如果超过了一周，则构造PATCH请求来关闭该issue
    data = {"state": "closed"}
    response = re.patch(issue_url, headers=headers, json=data)

    # 检查响应状态码
    if response.ok:
        print(f"Issue #{issue_number} closed successfully.")
    else:
        print(f"Error closing issue #{issue_number}: {response.text}")


if __name__ == "__main__":
    issues_data = {"title": "💡 {today} 来自OccCource更新提醒", "body": "{content}"}
    pull_issues(issues_data)
