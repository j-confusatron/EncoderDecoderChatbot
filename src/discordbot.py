import discord
from dotenv import load_dotenv
import loadData
import main
import modelconfigs
import os
import training

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

client = discord.Client()
guild = None

searcher, voc, maxLength = main.model(modelconfigs.Lstm_Dot_TopP_2_2_1500_ML20, False, False, False, True, False)

@client.event
async def on_ready():
    guild = discord.utils.get(client.guilds, name=GUILD)
    print(
        f'{client.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.channel.name == "therealinn" and message.content.lower().startswith("innbot!"):
        input_sentence = loadData.normalizeString(message.content[7:])
        output_words = training.callAndResponse(searcher, voc, input_sentence, maxLength)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        await message.channel.send(' '.join(output_words))

client.run(TOKEN)