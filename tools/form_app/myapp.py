#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json
import logging
import smtplib

# from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from flask import Flask, render_template, request

# from gevent import pywsgi

# from pull_issues from pull_issues

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # noqa:E501
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 发件人邮箱
mail_sender = "1144262839@qq.com"
# 邮箱授权码,注意这里不是邮箱密码,如何获取邮箱授权码,请看本文最后教程
# mail_license = os.getenv("MAIL_LICENSE")
mail_license = "szfiwfywaakhieda"
# 收件人邮箱，可以为多个收件人
mail_receivers = ["yidazhang1@gmail.com", "zhangwei@maiwei.ai"]


@app.route("/")
def index():
    return render_template("index.html")


def push_github():
    print("push_github")


def update_user_lists(user_name, email, zsxq_id, addition):
    # 打开文件并读取内容为字典
    with open("config.json", "r") as file:
        data = json.load(file)

    # 获取当前记录列表中最大的序号
    max_no = max(int(user["No."]) for user in data["users"])

    # 添加新记录
    new_record = {
        "No.": str(max_no + 1),  # 将序号设置为最大序号加1
        "name": user_name,
        "email": email,
        "zsxq_id": zsxq_id,
        "wechat_id": None,
        "isInWechatVipGroup": "",
        "feishu_id": None,
        "isNeedSourceCode": "",
        "github_id": None,
        "isStaredRepo": "",
        "phone_num": None,
        "addtion": addition,
    }

    data["users"].append(new_record)

    # 将更新后的字典写回文件
    with open("config.json", "w") as file:
        json.dump(data, file, indent=4)


def send_email(email_subject, mail_receivers, MIMEText_content):
    # 创建SMTP对象
    if "qq" in mail_sender:
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
    elif "gmail" in mail_sender:
        server = smtplib.SMTP("smtp.gmail.com", 587)  # Connect to the server
        server.starttls()
    elif "163" in mail_sender:
        server = smtplib.SMTP()
        # 设置发件人邮箱的域名和端口，端口地址为25
        server.connect("smtp.163.com", 25)
    else:
        print("Please check your sender email.")

    # Connect and login to the email server
    server.login(mail_sender, mail_license)

    if mail_receivers is None:
        logger.error("mail_receivers is None.")
        return False

    # Loop over each email to send to
    for mail_receiver in mail_receivers:
        # Setup MIMEMultipart for each email address (if we don't do this,
        # the emails will concatenate on each email sent)
        msg = MIMEMultipart()
        msg["From"] = mail_sender
        msg["To"] = mail_receiver
        msg["Subject"] = email_subject

        name = mail_receiver.split("@")[0]
        # 仅保留字母、数字和下划线字符
        name = "".join([i for i in name if i.isalnum() or i == "_"])
        print("Send to: ", name)

        # Attach the message to the MIMEMultipart object
        msg.attach(MIMEText_content)

        # Send the email to this specific email address
        server.sendmail(mail_sender, mail_receiver, msg.as_string())
        print("邮件发送成功！ 主题： {}".format(email_subject))

    # Quit the email server when everything is done
    server.quit()


@app.route("/submit", methods=["POST"])
def submit():
    username = request.form["username"]
    email = request.form["email"]
    zsxq_id = request.form["zsxq_id"]
    message = request.form["message"]

    # 发送邮件
    # sender_email = "1144262839@qq.com"
    # password = "szfiwfywaakhieda"
    receiver_email = ["yidazhang1@gmail.com"]
    subject = "New Form Submission"
    body = "Username: {}\nzsxq_id: {}\nEmail: {}\nMessage: {}".format(
        username, zsxq_id, email, message
    )
    print(body)

    mail_content = MIMEText(
        "Subject: {}\n\n{}".format(subject, body), "plain", "utf-8"
    )  # noqa:E501
    send_email(subject, receiver_email, mail_content)
    update_user_lists(username, email, zsxq_id, message)
    push_github()

    return "Form submitted successfully!"


if __name__ == "__main__":
    # server = pywsgi.WSGIServer(('0.0.0.0',5000),app)
    # server.serve_forever()
    app.run(host="0.0.0.0", port=5000)
