import discord
from discord.ext import commands
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env.local")
import os
import json

# Set up bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True  # Needed to read message content
intents.guilds = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Event for when bot is ready
@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")

    # Iterate through all guilds (servers) the bot is in
    for guild in bot.guilds:
        print(f"Fetching channels in: {guild.name}")
        
        # Check if the guild is the required one
        if guild.name == "Mr.Nobody's server":
            
            # Iterate through all channels in the guild
            for channel in guild.text_channels:
                # Create a filename based on the channel's name
                filename = f"{channel.name}.json"

                # Initialize a list to store messages
                messages = []

                print(f"Collecting messages from: {channel.name}")

                # Fetch message history
                async for message in channel.history(limit=None):  # Set limit=None to fetch all messages
                    # Append message data to the list as a dictionary
                    messages.append({
                        "timestamp": message.created_at.isoformat(),
                        "author": str(message.author),
                        "content": message.content
                    })

                # Write the messages to a JSON file
                with open(filename, "w", encoding="utf-8") as json_file:
                    json.dump(messages, json_file, indent=4, ensure_ascii=False)

                print(f"Saved chat history for channel '{channel.name}' to {filename}")

    await bot.close()  # Close the bot when done
    
# Run the bot
bot.run(os.getenv("DISCORD_BOT_API_KEY"))
