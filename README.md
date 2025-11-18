GitHub Readme Stats
GitHub Readme Stats
Get dynamically generated GitHub stats on your READMEs!

Tests Passing GitHub Contributors Tests Coverage Issues GitHub pull requests OpenSSF Scorecard



View Demo · Report Bug · Request Feature · FAQ · Ask Question

Love the project? Please consider donating to help it improve!

Table of contents (Click to show)
Important Notices
Important

Please note that we are NOT participating in Hacktoberfest this year. As a small team, we cannot handle the increased volume of PRs and issues that the event brings. We appreciate your understanding.

Important

Since the GitHub API only allows 5k requests per hour per user account, the public Vercel instance hosted on https://github-readme-stats.vercel.app/api could possibly hit the rate limiter (see #1471). We use caching to prevent this from happening (see https://github.com/anuraghazra/github-readme-stats#common-options). You can turn off these rate limit protections by deploying your own Vercel instance.

Uptime Badge

Important

We're a small team, and to prioritize, we rely on upvotes 👍. We use the Top Issues dashboard for tracking community demand (see #1935). Do not hesitate to upvote the issues and pull requests you are interested in. We will work on the most upvoted first.

GitHub Stats Card
Copy and paste this into your markdown, and that's it. Simple!

Change the ?username= value to your GitHub username.

[![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=anuraghazra)](https://github.com/anuraghazra/github-readme-stats)
Warning

By default, the stats card only shows statistics like stars, commits, and pull requests from public repositories. To show private statistics on the stats card, you should deploy your own instance using your own GitHub API token.

Note

Available ranks are S (top 1%), A+ (12.5%), A (25%), A- (37.5%), B+ (50%), B (62.5%), B- (75%), C+ (87.5%) and C (everyone). This ranking scheme is based on the Japanese academic grading system. The global percentile is calculated as a weighted sum of percentiles for each statistic (number of commits, pull requests, reviews, issues, stars, and followers), based on the cumulative distribution function of the exponential and the log-normal distributions. The implementation can be investigated at src/calculateRank.js. The circle around the rank shows 100 minus the global percentile.

Hiding individual stats
You can pass a query parameter &hide= to hide any specific stats with comma-separated values.

Options: &hide=stars,commits,prs,issues,contribs

![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=anuraghazra&hide=contribs,prs)
Showing additional individual stats
You can pass a query parameter &show= to show any specific additional stats with comma-separated values.

Options: &show=reviews,discussions_started,discussions_answered,prs_merged,prs_merged_percentage

![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=anuraghazra&show=reviews,discussions_started,discussions_answered,prs_merged,prs_merged_percentage)
Showing icons
To enable icons, you can pass &show_icons=true in the query param, like so:

![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=anuraghazra&show_icons=true)
Showing commits count for specified year
You can specify a year and fetch only the commits that were made in that year by passing &commits_year=YYYY to the parameter.

![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=anuraghazra&commits_year=2020)
Themes
With inbuilt themes, you can customize the look of the card without doing any manual customization.

Use &theme=THEME_NAME parameter like so :

![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=anuraghazra&show_icons=true&theme=radical)
All inbuilt themes
GitHub Readme Stats comes with several built-in themes (e.g. dark, radical, merko, gruvbox, tokyonight, onedark, cobalt, synthwave, highcontrast, dracula).

GitHub Readme Stats Themes

You can look at a preview for all available themes or checkout the theme config file. Please note that we paused the addition of new themes to decrease maintenance efforts; all pull requests related to new themes will be closed.

Responsive Card Theme
Anurag's GitHub stats-Dark

Since GitHub will re-upload the cards and serve them from their CDN, we can not infer the browser/GitHub theme on the server side. There are, however, four methods you can use to create dynamics themes on the client side.

Use the transparent theme
We have included a transparent theme that has a transparent background. This theme is optimized to look good on GitHub's dark and light default themes. You can enable this theme using the &theme=transparent parameter like so:

![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=anuraghazra&show_icons=true&theme=transparent)
👀 Show example
Add transparent alpha channel to a themes bg_color
You can use the bg_color parameter to make any of the available themes transparent. This is done by setting the bg_color to a color with a transparent alpha channel (i.e. bg_color=00000000):

![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=anuraghazra&show_icons=true&bg_color=00000000)
👀 Show example
Use GitHub's theme context tag
You can use GitHub's theme context tags to switch the theme based on the user GitHub theme automatically. This is done by appending #gh-dark-mode-only or #gh-light-mode-only to the end of an image URL. This tag will define whether the image specified in the markdown is only shown to viewers using a light or a dark GitHub theme:

[![Anurag's GitHub stats-Dark](https://github-readme-stats.vercel.app/api?username=anuraghazra&show_icons=true&theme=dark#gh-dark-mode-only)](https://github.com/anuraghazra/github-readme-stats#gh-dark-mode-only)
[![Anurag's GitHub stats-Light](https://github-readme-stats.vercel.app/api?username=anuraghazra&show_icons=true&theme=default#gh-light-mode-only)](https://github.com/anuraghazra/github-readme-stats#gh-light-mode-only)
👀 Show example
Use GitHub's new media feature
You can use GitHub's new media feature in HTML to specify whether to display images for light or dark themes. This is done using the HTML <picture> element in combination with the prefers-color-scheme media feature.

<picture>
  <source
    srcset="https://github-readme-stats.vercel.app/api?username=anuraghazra&show_icons=true&theme=dark"
    media="(prefers-color-scheme: dark)"
  />
  <source
    srcset="https://github-readme-stats.vercel.app/api?username=anuraghazra&show_icons=true"
    media="(prefers-color-scheme: light), (prefers-color-scheme: no-preference)"
  />
  <img src="https://github-readme-stats.vercel.app/api?username=anuraghazra&show_icons=true" />
</picture>
👀 Show example
Customization
You can customize the appearance of all your cards however you wish with URL parameters.

Common Options
Name	Description	Type	Default value
title_color	Card's title color.	string (hex color)	2f80ed
text_color	Body text color.	string (hex color)	434d58
icon_color	Icons color if available.	string (hex color)	4c71f2
border_color	Card's border color. Does not apply when hide_border is enabled.	string (hex color)	e4e2e2
bg_color	Card's background color.	string (hex color or a gradient in the form of angle,start,end)	fffefe
hide_border	Hides the card's border.	boolean	false
theme	Name of the theme, choose from all available themes.	enum	default
cache_seconds	Sets the cache header manually (min: 21600, max: 86400).	integer	21600
locale	Sets the language in the card, you can check full list of available locales here.	enum	en
border_radius	Corner rounding on the card.	number	4.5
Warning

We use caching to decrease the load on our servers (see #1471 (comment)). Our cards have the following default cache hours: stats card - 24 hours, top languages card - 144 hours (6 days), pin card - 240 hours (10 days), gist card - 48 hours (2 days), and wakatime card - 24 hours. If you want the data on your cards to be updated more often you can deploy your own instance and set environment variable CACHE_SECONDS to a value of your choosing.

Gradient in bg_color
You can provide multiple comma-separated values in the bg_color option to render a gradient with the following format:

&bg_color=DEG,COLOR1,COLOR2,COLOR3...COLOR10
Available locales
Here is a list of all available locales:

Code	Locale
ar	Arabic
az	Azerbaijani
bn	Bengali
bg	Bulgarian
my	Burmese
ca	Catalan
cn	Chinese
zh-tw	Chinese (Taiwan)
cs	Czech
nl	Dutch
en	English
fil	Filipino
fi	Finnish
fr	French
de	German
el	Greek
Code	Locale
he	Hebrew
hi	Hindi
hu	Hungarian
id	Indonesian
it	Italian
ja	Japanese
kr	Korean
ml	Malayalam
np	Nepali
no	Norwegian
fa	Persian (Farsi)
pl	Polish
pt-br	Portuguese (Brazil)
pt-pt	Portuguese (Portugal)
ro	Romanian
Code	Locale
ru	Russian
sa	Sanskrit
sr	Serbian (Cyrillic)
sr-latn	Serbian (Latin)
sk	Slovak
es	Spanish
sw	Swahili
se	Swedish
ta	Tamil
th	Thai
tr	Turkish
uk-ua	Ukrainian
ur	Urdu
uz	Uzbek
vi	Vietnamese
If we don't support your language, please consider contributing! You can find more information about how to do it in our contributing guidelines.

Stats Card Exclusive Options
Name	Description	Type	Default value
hide	Hides the specified items from stats.	string (comma-separated values)	null
hide_title	Hides the title of your stats card.	boolean	false
card_width	Sets the card's width manually.	number	500px  (approx.)
hide_rank	Hides the rank and automatically resizes the card width.	boolean	false
rank_icon	Shows alternative rank icon (i.e. github, percentile or default).	enum	default
show_icons	Shows icons near all stats.	boolean	false
include_all_commits	Count total commits instead of just the current year commits.	boolean	false
line_height	Sets the line height between text.	integer	25
exclude_repo	Excludes specified repositories.	string (comma-separated values)	null
custom_title	Sets a custom title for the card.	string	<username> GitHub Stats
text_bold	Uses bold text.	boolean	true
disable_animations	Disables all animations in the card.	boolean	false
ring_color	Color of the rank circle.	string (hex color)	2f80ed
number_format	Switches between two available formats for displaying the card values short (i.e. 6.6k) and long (i.e. 6626).	enum	short
number_precision	Enforce the number of digits after the decimal point for short number format. Must be an integer between 0 and 2. Will be ignored for long number format.	integer (0, 1 or 2)	null
show	Shows additional items on stats card (i.e. reviews, discussions_started, discussions_answered, prs_merged or prs_merged_percentage).	string (comma-separated values)	null
commits_year	Filters and counts only commits made in the specified year.	integer (YYYY)	<current year> (one year to date)
Warning

Custom title should be URI-escaped, as specified in Percent Encoding (i.e: Anurag's GitHub Stats should become Anurag%27s%20GitHub%20Stats). You can use urlencoder.org to help you do this automatically.

Note

When hide_rank=true, the minimum card width is 270 px + the title length and padding.

GitHub Extra Pins
GitHub extra pins allow you to pin more than 6 repositories in your profile using a GitHub readme profile.

Yay! You are no longer limited to 6 pinned repositories.

Usage
Copy-paste this code into your readme and change the links.

Endpoint: api/pin?username=anuraghazra&repo=github-readme-stats

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=anuraghazra&repo=github-readme-stats)](https://github.com/anuraghazra/github-readme-stats)
Options
You can customize the appearance and behavior of the pinned repository card using the common options and exclusive options listed in the table below.

Name	Description	Type	Default value
show_owner	Shows the repo's owner name.	boolean	false
description_lines_count	Manually set the number of lines for the description. Specified value will be clamped between 1 and 3. If this parameter is not specified, the number of lines will be automatically adjusted according to the actual length of the description.	number	null
Demo
Readme Card

Use show_owner query option to include the repo's owner username

Readme Card

GitHub Gist Pins
GitHub gist pins allow you to pin gists in your GitHub profile using a GitHub readme profile.

Usage
Copy-paste this code into your readme and change the links.

Endpoint: api/gist?id=bbfce31e0217a3689c8d961a356cb10d

[![Gist Card](https://github-readme-stats.vercel.app/api/gist?id=bbfce31e0217a3689c8d961a356cb10d)](https://gist.github.com/Yizack/bbfce31e0217a3689c8d961a356cb10d/)
Options
You can customize the appearance and behavior of the gist card using the common options and exclusive options listed in the table below.

Name	Description	Type	Default value
show_owner	Shows the gist's owner name.	boolean	false
Demo
Gist Card

Use show_owner query option to include the gist's owner username

Gist Card
