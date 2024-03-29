from flask import Flask, render_template, request
from github import Github

app = Flask(__name__)

# 配置GitHub访问令牌
github_token = "YOUR_GITHUB_TOKEN"
github_repo = "YOUR_REPO"


@app.route("/")
def index():
    return render_template("form.html")


@app.route("/submit", methods=["POST"])
def submit():
    username = request.form["username"]
    email = request.form["email"]
    message = request.form["message"]

    # 创建GitHub Issue
    g = Github(github_token)
    repo = g.get_repo(github_repo)
    issue = repo.create_issue(
        title="新的表单提交", body=f"用户名: {username}\n邮箱: {email}\n备注: {message}"
    )

    return f"表单提交成功！创建的Issue链接: {issue.html_url}"


if __name__ == "__main__":
    app.run()
