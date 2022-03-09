
import smtplib
import imghdr
from email.message import EmailMessage

EMAIL_ADDRESS = 'nibogoji@gmail.com'
EMAIL_PASSWORD = 'stefa0821!'

contacts = ['nibogoji@gmail.com']

msg = EmailMessage()
msg['Subject'] = 'Test'
msg['From'] = EMAIL_ADDRESS
msg['To'] = 'nibogoji@gmail.com'

msg.set_content('blablabla')

msg.add_alternative("""\
<!DOCTYPE html>
<html>
    <body>
        <h1 style="color:SlateGray;">This is an HTML Email!</h1>
    </body>
</html>
""", subtype='html')


with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    smtp.send_message(msg)