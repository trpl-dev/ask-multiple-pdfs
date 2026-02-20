css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
"""

# Avatars are self-contained SVG data URIs — no external network requests.
_BOT_AVATAR = (
    "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9Ij"
    "AgMCAxMDAgMTAwIj48Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSI1MCIgZmlsbD0iIzRDQUY1MCIvPjxyZWN0IHg9"
    "IjI1IiB5PSIzMCIgd2lkdGg9IjUwIiBoZWlnaHQ9IjQwIiByeD0iOCIgZmlsbD0id2hpdGUiLz48Y2lyY2xlIGN4PS"
    "IzOCIgY3k9IjQ3IiByPSI3IiBmaWxsPSIjNENBRjUwIi8+PGNpcmNsZSBjeD0iNjIiIGN5PSI0NyIgcj0iNyIgZmls"
    "bD0iIzRDQUY1MCIvPjxyZWN0IHg9IjM1IiB5PSI2MiIgd2lkdGg9IjMwIiBoZWlnaHQ9IjQiIHJ4PSIyIiBmaWxsPSI"
    "jNENBRjUwIi8+PHJlY3QgeD0iNDYiIHk9IjE2IiB3aWR0aD0iOCIgaGVpZ2h0PSIxNCIgZmlsbD0id2hpdGUiLz48L3N2Zz4="
)

_USER_AVATAR = (
    "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9Ij"
    "AgMCAxMDAgMTAwIj48Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSI1MCIgZmlsbD0iIzIxOTZGMyIvPjxjaXJjbGUgY"
    "3g9IjUwIiBjeT0iMzgiIHI9IjE4IiBmaWxsPSJ3aGl0ZSIvPjxlbGxpcHNlIGN4PSI1MCIgY3k9Ijg1IiByeD0iMjgi"
    "IHJ5PSIyMiIgZmlsbD0id2hpdGUiLz48L3N2Zz4="
)

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="{_BOT_AVATAR}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
'''

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="{_USER_AVATAR}">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
'''
