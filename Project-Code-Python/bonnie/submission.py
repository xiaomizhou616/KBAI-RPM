from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import input
from builtins import object
import os
import sys
import zipfile
import json
import re
import getpass
import errno
import base64
import requests
from urllib.parse import urlsplit

class BonnieAuthenticationError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


def default_app_data_dir():
  APPNAME = "bonnie"
  
  if sys.platform == 'win32':
    return os.path.join(os.environ['APPDATA'], APPNAME)
  else:
    return os.path.expanduser(os.path.join("~", "." + APPNAME))



class Submission(object):
  url = {'local': 'http://localhost:3000',
         'development': 'https://bonnie-dev.udacity.com',
         'staging': 'https://bonnie-staging.udacity.com',
         'production': 'https://bonnie.udacity.com'}

  submission_filename = 'student.zip'
  
  def __init__(self, gtcode, quiz_name, 
                filenames = [], 
                exclude = False, 
                environment = 'production', 
                provider = 'gt',
                app_data_dir = None,
                max_zip_size = 8388608):

    self.gtcode = gtcode
    self.quiz_name = quiz_name
    self.filenames = filenames
    self.exclude = exclude
    self.provider = provider
    self.app_data_dir = app_data_dir or default_app_data_dir()
    self.max_zip_size = max_zip_size

    self.bonnie_url = Submission.url[environment]
    self.udacity_url = "https://www.udacity.com"

    self.jwt_path = os.path.join(self.app_data_dir, "jwt")

    if self.exclude:
      raise ValueError("Exclude is no longer supported as an argument.")

    self._authorize_session()

    self.submit_url = self._get_submit_url()

    self._mkzip()

    with open("student.zip", "rb") as fd:
      data = {"zipfile": base64.b64encode(fd.read()).decode('ascii')}

    try:
      self.r = self.s.post(self.submit_url, 
                           data=json.dumps(data))
      self.r.raise_for_status()
    except requests.exceptions.HTTPError as e:
      if self.r.status_code == 403:
        raise RuntimeError("You don't have access to this quiz.")
      elif self.r.status_code in [404,500]:
        message = self.r.json()["message"]
        raise RuntimeError(message)
      else:
        raise

    self.submission = self.r.json()

  def poll(self):
    self.r = self.s.get(self._get_poll_url())
    self.r.raise_for_status()

    self.submission = self.r.json()

    return self.submission['feedback'] is not None or self.submission['error_report'] is not None

  def result(self):
    return self.feedback()

  def feedback(self):
    return self.submission['feedback']

  def error_report(self):
    return self.submission['error_report']

  def _set_auth_headers(self, jwt):
    self.s.headers.update({'authorization': 'Bearer ' + jwt})

  def _bonnie_login(self):
    try:
      if self.provider == 'udacity':
        print("Udacity Login required.")
        username = input('Email :')
        password = getpass.getpass('Password :') 

        data = {'udacity' : {'username' : username, 'password' : password}}

        #Logging into udacity
        r = self.s.post(self.udacity_url + '/api/session', data=json.dumps(data))
        r.raise_for_status()     

        #Logging into bonnie
        r = self.s.get(self.bonnie_url + '/auth/udacity')  
        r.raise_for_status()

      elif self.provider == 'gt':
        print("GT Login required.")
        username = input('Username :')
        password = getpass.getpass('Password :')

        r = self.s.get(self.bonnie_url + '/auth/cas',
                       headers = {'accept': '*/*'})
        r.raise_for_status

        host = '://'.join(urlsplit(r.url)[0:2])

        action, data = self._scrape_gt_auth(r.text)

        data['username'] = username
        data['password'] = password

        r = self.s.post(host + action, data=data, 
                        headers = {'content-type': 'application/x-www-form-urlencoded', 'accept': '*/*'})
        r.raise_for_status()

        if not r.url.startswith("https://bonnie.udacity.com"):
          raise ValueError("Username and password failed (Do you use two-factor?)")

    except requests.exceptions.HTTPError as e:
      if e.response.status_code == 403:
        raise BonnieAuthenticationError("Authentication failed")
      else:
        raise e

    #Checking that login worked
    r = self.s.get(self.bonnie_url + '/users/me')
    r.raise_for_status()

    #Acquiring auth token for future use
    r = self.s.post(self.bonnie_url + '/auth_tokens')
    r.raise_for_status()

    jwt = r.json()['auth_token']

    self._set_auth_headers(jwt)

    save = input('Save the jwt?[y,N]')
    if save.lower() == 'y':
      try:
        os.makedirs(self.app_data_dir)
      except OSError as exception:
        if exception.errno != errno.EEXIST:
          raise

      try:
        with open(self.jwt_path, "r") as fd:
            jwt_obj = json.load(fd)
      except:
        jwt_obj = {}

      jwt_obj[self.provider] = jwt
      with open(self.jwt_path, "w") as fd:
        json.dump(jwt_obj, fd)

  def _authorize_session(self):
    self.s = requests.Session()
    self.s.headers.update({'content-type':'application/json;charset=UTF-8', 'accept': 'application/json'})

    try:
      with open(self.jwt_path, "r") as fd:
        jwt_obj = json.load(fd)

      self._set_auth_headers(jwt_obj[self.provider])

      r = self.s.get(self.bonnie_url + "/users/me")
      r.raise_for_status()
    except (requests.exceptions.HTTPError, IOError, ValueError, KeyError) as e:
      self._bonnie_login()

  def _get_submit_url(self):
    return self.bonnie_url + "/student/course/%s/quiz/%s/submission" % (self.gtcode, self.quiz_name)   

  def _get_poll_url(self):
    return self.bonnie_url + "/student/course/%s/quiz/%s/submission/%s" % (self.gtcode, self.quiz_name, self.submission['id'])

  def _mkzip(self):
    filenames = [os.path.normpath(x) for x in self.filenames]

    dirname = os.path.dirname(sys.argv[0])

    if os.path.commonprefix([dirname] + filenames) != dirname:
      raise ValueError("Submitted files must in subdirectories of %s." % base)

    with zipfile.ZipFile(Submission.submission_filename,'w') as z:
      for f in self.filenames:
        z.write(f, os.path.relpath(f, dirname))

    if os.stat(Submission.submission_filename).st_size > self.max_zip_size:
      raise ValueError("Your zipfile exceeded the limit of %d bytes" % self.max_zip_size)

  def _scrape_gt_auth(self, text):
    action = re.search('action="([^"]*)" method="post">', text).group(1)
    lt = re.search('<input type="hidden" name="lt" value="([^"]*)" />', text).group(1)
    execution = re.search('<input type="hidden" name="execution" value="([^"]*)" />', text).group(1)
    _eventId = re.search('<input type="hidden" name="_eventId" value="([^"]*)" />', text).group(1)
    warn = False

    return action, {'lt': lt, 'execution': execution, '_eventId': _eventId, 'warn': warn}
