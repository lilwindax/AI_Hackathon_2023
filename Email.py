# Library includes
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


# Function to send emails
def Send_Email(email_address, detection_image):
    # Variable initialization
    fromaddr = "testtargetemail121362472357@gmail.com"
    toaddr = email_address

    # Initialize MINEMutilpart
    msg = MIMEMultipart()

    # Configure email
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Detection Notification"

    body = "Dear user there has been a detection. Please see attachment below: "

    msg.attach(MIMEText(body, 'plain'))

    # Load detected image as attachment to email
    filename = detection_image
    attachment = open(detection_image, "rb")

    # Configuration
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)

    # Send email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "pabumnvgcdzhipsy")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()