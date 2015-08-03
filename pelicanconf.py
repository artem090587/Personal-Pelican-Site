#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals


AUTHOR = 'Bryan Smith'
SITENAME = "Bryan Travis Smith, Ph.D"
ALT_NAME = SITENAME
SITESUBTITLE = "Physicist, Data Scientist, Martial Artist, & Life Enthusiast"
SITE_SUBTEXT = SITESUBTITLE
DESCRIPTION = "A blog about my experience with physics, data science, artifical intelligences, robotics, and technology."
SITEURL = 'http://www.bryantravissmith.com'
#FAVICON = 'favicon.ico'
#FAVICON_TYPE = 'image/vnd.microsoft.icon'

META_IMAGE = SITEURL + "/img/bryan.jpeg"
META_IMAGE_TYPE = "image/jpeg"
ARTICLE_PATHS = ['blog']
PAGE_PATHS = ['pages']
PATH = 'content'
STATIC_PATHS = ['img', 'files','udacity','data-vis']
TIMEZONE = 'America/Los_Angeles'

ARTICLE_URL = '{category}/{slug}/'
ARTICLE_SAVE_AS = '{category}/{slug}/index.html'
PAGE_URL = '{slug}/'
PAGE_SAVE_AS = '{slug}/index.html'

DEFAULT_LANG = 'en'
LOCALE = 'en_GB'

THEME = "alchemy-theme/"

EMAIL_ADDRESS = 'bryantravissmith@gmail.com'
GITHUB_ADDRESS = 'https://github.com/bryantravissmith'

# Social widget
#SOCIAL = (('linkedin','https://linkedin.com/in/bryantravissmith'),
#		  ('github', 'https://github.com/bryantravissmith'),
#          ('google-plus-square', 'https://plus.google.com/+BryanSmithPhD'),
#		  ('twitter', 'https://twitter.com/bryantravissmit'),
#          ('instagram','https://instagram.com/bryantravissmith/'),
#          ('envelope', 'mailto:bryantravissmith@gmail.com'),)

#SHARE = True

#FOOTER = ("&copy; 2015 Bryan Smith. All rights reserved.<br>" +
#          "Code snippets in the pages are released under " +
#          "<a href=\"http://opensource.org/licenses/MIT\" target=\"_blank\">" +
#          "The MIT License</a>, unless otherwise specified.")

PAGES_ON_MENU = True
CATEGORIES_ON_MENU = True
SHOW_ARTICLE_AUTHOR = True
TAGS_ON_MENU = True

PROFILE_IMAGE = "http://www.bryantravissmith.com/img/bryan.jpeg"

DEFAULT_PAGINATION = 10

TAG_SAVE_AS = ''
AUTHOR_SAVE_AS = ''
#DIRECT_TEMPLATES = ('index', 'categories', 'archives', 'search', 'tipue_search')
#TIPUE_SEARCH_SAVE_AS = 'tipue_search.json'

RELATIVE_URLS = False

FEED_ALL_ATOM = 'feeds/all.atom.xml'

DELETE_OUTPUT_DIRECTORY = True

#TWITTER_USERNAME = 'bryantravissmit'
DISQUS_SITENAME = "bryansmithphd"
#SC_PROJECT = '10224955'
#SC_SECURITY = '1f2cc438'
GOOGLE_ANALYTICS_ID = "UA-24340005-3"
GOOGLE_ANALYTICS_DOMAIN = "www.bryantravissmith.com"

