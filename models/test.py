import smtplib

SENDER_EMAIL = "aadithyanlearn@gmail.com"
RECEIVER_EMAIL = "aadithyanpramad@gmail.com"


subject="HI"
message="Bye"

text=f"Subject:{subject}\n\n{message}"

server=smtplib.SMTP("smtp.gmail.com",587)
server.starttls()

server.login(SENDER_EMAIL,"lsqn oulw djuv ltvt")

server.sendmail(SENDER_EMAIL,RECEIVER_EMAIL,text)
print("Email has been sent to "+RECEIVER_EMAIL)