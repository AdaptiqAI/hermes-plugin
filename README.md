# AdaptiqAI for Hermes

Use your AdaptiqAI workspace's models in Hermes. You sign in once in the browser, so there is no
key to copy or paste. Each Hermes conversation keeps its own native memory on AdaptiqAI.

## Install

- **Hermes Desktop:** open `hermes://plugin/install?repo=AdaptiqAI/hermes-plugin`, or use the
  "Install in Hermes" button in the AdaptiqAI console. Click Install. Sign-in starts on its own,
  and your browser asks you to approve the connection.
- **Terminal:** `hermes plugins install AdaptiqAI/hermes-plugin`, then `hermes adaptiq signin`.
- **Connect to app** in the AdaptiqAI console sets all of it up from one command.

Hermes doesn't update a plugin by itself. Once a day the plugin reads the version the mirror publishes, and when
it is newer, `/adaptiq` and a Desktop notice say so with the command: `hermes plugins update adaptiq`, then
restart Hermes.

## The /adaptiq command

`/adaptiq` answers the same way as in every harness AdaptiqAI has a plugin for:

| Command | What it does |
|---|---|
| `/adaptiq` or `/adaptiq status` | The connection, this conversation (its memory, whether it is saved, the knowledge it uses), the library and conversation memory, each model with its window and how many conversations run on it now, and the main model. Warnings come last: memory nearly full, a full library, idle conversations sleeping, a connection about to end. Then where the conversation's memory is, if it now runs on another model, and a newer plugin with the command that installs it. |
| `/adaptiq signin` | Connects your AdaptiqAI account in the browser. |
| `/adaptiq private` / `/adaptiq remember` | Stops or resumes saving this conversation's new messages. |
| `/adaptiq knowledge` | The knowledge this conversation uses, and the collections ready for its model. |
| `/adaptiq use <name>` | Uses a collection here, or an earlier conversation on the same model, read from memory. |
| `/adaptiq stop <name>` | Stops using it here. Nothing is deleted. |
| `/adaptiq open` | Opens this conversation in AdaptiqAI Knowledge. |
| `/adaptiq diagnostics` | A report to send to support. It holds no keys. |
| `/adaptiq signout` | Forgets this computer's sign-in. |

`use` and `stop` take an exact name first, else the only name that contains what you typed. They never guess
between several. On a messaging platform, or a computer without a graphical browser, a link is shown instead of
opened.

In a terminal: `hermes adaptiq` (the status), `hermes adaptiq signin` and `hermes adaptiq signout`.

## Hermes Desktop

- **The status bar** shows "AdaptiqAI · 25k / 5M" for the conversation in focus while it runs on an AdaptiqAI
  model: how much of it memory holds, against the model's window. It adds "private" when nothing new is saved. For
  a moment after a reply it says what the reply read from memory ("+24.6k from memory", with "· free" when reading
  from memory costs nothing). A click opens a panel with:
  - Remember and Private, like `/adaptiq remember` and `/adaptiq private`;
  - the collections the conversation uses, and the ready ones for its model, each used or stopped in one click,
    with a button to AdaptiqAI Knowledge;
  - what the last turn and the whole conversation processed: steps, new input, read from memory, saved to memory,
    output and cost, counted on this computer;
  - the button a refusal asks for, until a request goes through: Sign in again, Add funds or Get access.
- **Settings → AdaptiqAI** shows:
  - the workspace, marked "(API key)" when the connection is an API key;
  - the memory, and how much of it is conversations and how much knowledge;
  - each model and how many conversations run on it now, with the main model marked;
  - in its last two weeks, when the connection ends.

  It has buttons to sign in (again), to sign out and to open AdaptiqAI Knowledge.
- **Notices:** memory nearly full (from 90 %), a full memory or library, and idle conversations put to sleep
  are each said once, with a button to the console. A newer plugin is said once, with the command that
  installs it. A refusal you can end is said once, with its button: Sign in again, Add funds or Get access.

In a terminal or a chat platform, the first reply on AdaptiqAI after a memory warning or a sleeping conversation
appears ends with it, once; a warning is said again when it rises.

## Signing in and out

- **Desktop:** Settings → AdaptiqAI, or "AdaptiqAI: Sign in" / "AdaptiqAI: Sign out" in the command palette.
- **Any chat:** `/adaptiq signin` and `/adaptiq signout`.
- **Terminal:** `hermes adaptiq signin` and `hermes adaptiq signout`.

A connection lasts 90 days. You can see it in the console under **API keys**, as "Hermes on <your computer>",
and revoking it there disconnects this computer only.

The key goes into Hermes' `.env` as `ADAPTIQ_API_KEY`, which Hermes sends, and as `ADAPTIQ_SIGNIN_KEY`, this
computer's own. Connect to app writes `ADAPTIQ_API_KEY` too. The plugin tries every key Hermes has, in turn:
`ADAPTIQ_API_KEY` as `.env` holds it, as a running Hermes holds it, the sign-in key, then the key of any other
AdaptiqAI endpoint. A key AdaptiqAI refuses gives way to the next. When the refused key is the one Hermes sends
and another works, `ADAPTIQ_API_KEY` gets the working key.

Signing out removes this computer's sign-in key, and `ADAPTIQ_API_KEY` too when it holds that key. A key
Connect set up stays, so Hermes stays connected with it, as an API key.

Without a browser (a server, CI), set `ADAPTIQ_API_KEY` to an API key from the console. Nothing else is needed.

## Models

Your workspace's models appear in Hermes' model list, and `/adaptiq` lists them. You pay per token from the
workspace balance, and reads from memory are free. Lanes and plans are no longer sold.

Each request follows the model's capability card:

- **Thinking:** Hermes' level is sent the way the model names it. A level the model lacks becomes its nearest
  one, and Hermes' off becomes the model's off. When you have chosen no level in Hermes, the model's own
  default is used.
- **Output:** the most a reply may be is the model's own cap.
- **Images:** go only to a model that reads them. For one that reads text only, Hermes describes them with
  its vision model.

Hermes counts what a reply read from memory as cached input. Its own cost counter prices each request at the
model's prices: new input, output, what is read from memory and what is saved to it. For that the plugin gives the
AdaptiqAI entry a provider profile of its own, which is Hermes' generic one in every other way and is hidden from
its pickers: Hermes' own reading of the prices stops at the model's capability card, which comes first in its row.

## Memory per conversation

Every turn of a Hermes conversation tells AdaptiqAI which conversation it belongs to, using Hermes'
own session id. So AdaptiqAI keeps that conversation's memory, under the same id when you resume it.

AdaptiqAI keeps the whole conversation, so it never needs compacting. Hermes has no way for a plugin to
decline compaction for one provider, so the plugin does two things:

- **Compaction only when the conversation no longer fits.** Hermes compacts AdaptiqAI models only once
  their input budget is full, where the request would not fit anyway. Other providers keep your
  settings.
- **One memory per conversation, compacted or not.** Hermes compacts in place by default and keeps the
  session id. Where it starts a new session id instead (`compression.in_place: false`, or an older
  Hermes), the continuation still uses the memory of the conversation it continues. It does not start
  a second one beside it. A subagent keeps a conversation of its own.
- **Subagents shown under their conversation.** A subagent's requests name the conversation that
  started it, so the AdaptiqAI console lists it there, as a helper. A Kanban worker names the conversation that
  made its task (when one did) as a track: a line of work of its own, linked to that conversation.
- **Memory that takes a while to load.** A turn asks the engine for progress frames while it prepares a large
  conversation's memory, so Hermes sees a live stream rather than a stalled one.
- **Switching models.** A conversation's memory stays on the AdaptiqAI model it started on. Switch it to another
  model with `/model`, and the first reply there ends with one line: where the memory is,
  `/model <id> --provider <entry>` to switch back, and `/new` to start fresh on the new model. It says so once per
  switch (again after Hermes restarts), and `/adaptiq` says it while the conversation runs elsewhere.

A `/compress` you run yourself still compacts. AdaptiqAI then rebuilds that conversation's memory from
the shorter history, under the same conversation.

Some requests are not a turn of a conversation and run once, saving nothing:

- Hermes' helper calls, such as titles, compression, approvals and vision.
- Its background reviews and `/btw` side questions, which replay a conversation.

You choose per conversation:

- `/adaptiq private` stops saving the new messages of the conversation you are in. It keeps the
  memory it already has.
- `/adaptiq remember` turns saving back on.
- `/adaptiq status` says which one applies.

This needs the plugin enabled. With it disabled, AdaptiqAI saves no Hermes conversation (an entry written by 0.1.0 gets the mark the next time the plugin loads).

## Knowledge

Collections from AdaptiqAI Knowledge can be used in a conversation. A collection is built for one model and
works with that model only, so the agent searches the store of the model the conversation runs on.

- **`adaptiq_knowledge`** (agent tool) says what the conversation uses: its collections and the conversations
  added to it. It finds words, ids or names in the text of those collections and gives the exact lines as
  document:line, lists their documents, and opens one. Asked to find in named documents (up to 12, or up to 1,000
  small ones of 1M characters in all: with commas between, by a pattern, or by the start or a part of their names),
  it reads the documents themselves, sixteen at a time, and lists
  every line that holds the words as one list, each with its record (a JSON object, a paragraph or a table's row),
  page by page; words that stand on different lines of one record find the records that hold them all. `fields`
  shows only the fields asked for, one record a line (`paid-due` gives the days between two dates, and a text's
  field may be the words a number stands next to in a sentence: "21 rejected units"); `sort` orders
  the records of all the documents together; `sum` adds fields up over every record found; and `group` gives one
  line a value, with its count and sums. A CSV or TSV is read from its original, so its cells and decimal commas
  are as written. It searches the library by collection name, and starts or
  stops using a collection by its name, which applies from the next reply; asked to use one that could not be built,
  it says why (a full workspace: how much is taken and how much it needs). A turn is told which ready collections
  its conversation reads, once and again when they change, so the agent looks things up there. status names the
  collections the earlier conversations added here read and this one does not, and find, with no collection in use
  here, says to use one of them first.
- **`adaptiq_memory`** (agent tool) reads the workspace's memory use and lists collections with their sizes. It
  opens AdaptiqAI Knowledge in your browser for cleaning up (or gives the link where no browser is at hand), and
  stops saving the conversation when you ask it to.
- **`adaptiq_create_knowledge`** (agent tool) makes a collection from text, files on this computer, or HTTPS
  links, which can be GitHub repositories. It does so only when you ask for it in your own words, or say yes to
  the agent's offer to save it (typed, or answered through clarify), and never from a subagent. Once the
  collection is ready, it is used in the conversation. For 15 minutes the agent can correct a collection it just
  saved (the same collection gets the new content; no second one is made) and, when you ask, take it back. It
  removes nothing else; clean-up happens in the console.
- **`/adaptiq knowledge`**, **`/adaptiq use <name>`** and **`/adaptiq stop <name>`** do the same from the chat.
- **`@adaptiq:`** in a message names an earlier conversation, which the turn then adds to this one, so the model
  reads it from memory. The TUI's and Desktop's `@` menu offer it and complete the earlier conversations on the
  chat's model; in the classic CLI you type the name, quoted when it has spaces. Only a conversation on the same
  model is added, nothing while the conversation is private, and it stays for later turns. The turn waits up to
  eight seconds for it to be ready, and the reply says what came of each mention.

The tools are the DeepSeek Harness plugin's, with the same descriptions and the same plain answers, which name
collections and never show ids. They are offered whenever AdaptiqAI is connected, whatever Hermes' main
provider is: Hermes gives a conversation its tools when it starts, so a chat moved onto AdaptiqAI with `/model`
has them too. In a conversation on another provider they say they work on an AdaptiqAI model.

A new conversation on an AdaptiqAI model gets the knowledge instructions in its system prompt: look a fact up
with find before answering, and cite the lines. Hermes freezes a conversation's system prompt when it starts.
A conversation that started before, or on another provider, gets them once, in its first turn on AdaptiqAI.

The start of a turn waits at most two seconds for AdaptiqAI. Checking new collections, and looking up what the
conversation reads, run beside the turn.

Hermes hides a plugin's tools behind its tool search (`tools.tool_search.enabled`, on by default), so the agent
sees them by name only and has to look them up. Hermes has no switch for one plugin's tools. Connect to app turns
tool search off for every tool, so the knowledge tool is always in view.

## When something goes wrong

The plugin tells Hermes what AdaptiqAI's answers mean, so Hermes recovers the right way:

- **A model not in the workspace, or one that stopped while idle.** Hermes says the model isn't available and
  that you can pick another with `/model`. It no longer tells you to fix your API key.
- **An empty balance.** Hermes stops and says to add funds, with the link to the console's billing page. It
  does not bench the key, so a model works again as soon as funds are in.
- **Memory full.** Hermes stops instead of retrying, and never compresses the conversation (AdaptiqAI
  keeps it whole). You can free up space in Knowledge, use another model, or continue with
  `/adaptiq private`. Where AdaptiqAI puts idle conversations to sleep to make room (their compiled
  memory is dropped, the transcript is kept, and they wake when you use them; nothing is deleted),
  a full memory means nothing idle could sleep, and Hermes says: "Memory is full and nothing idle
  could sleep. Delete or unpin a conversation in the console, or add memory." A new conversation
  frees nothing.
- **A busy model, or a conversation being prepared.** Hermes retries by itself, after the wait AdaptiqAI asks
  for (Retry-After).
- **Expired sign-in.** Hermes asks you to run `/adaptiq signin`.

Hermes Desktop shows these in AdaptiqAI's own words.

## What it changes in Hermes

- `ADAPTIQ_API_KEY` and `ADAPTIQ_SIGNIN_KEY` in Hermes' `.env`.
- A `providers.AdaptiqAI` entry pointing at `https://api.adaptiqai.com/v1`, with the key taken from
  that variable. The model list is read live from the workspace, so a model added later appears
  without signing in again. Each model gets its context window, and `supports_vision` from its card.
- The main model becomes the workspace model that can run the most conversations at once.
- The entry follows your workspace's models. The plugin brings it in step when a conversation starts (at most every
  ten minutes) and on `/adaptiq status`:
  - A model added later gets its context window.
  - A model that left the workspace leaves the entry (the engine's own model stays listed, and is never made the main model).
  - If Hermes' main model was one of those, it moves to the model that can run the most at once.
- `compression.model_thresholds` gets `0.99` for each AdaptiqAI model that has no threshold of its
  own, so Hermes compacts those conversations only when they no longer fit (see Memory per
  conversation). A threshold you set yourself is kept, except 1.0 or more: Hermes reads that as 85%
  of the window, so it becomes `0.99` too.
- The entry's `extra_body` marks requests to it as run-once (`adaptiq_native_memory`). Hermes' helper
  calls inherit that. The plugin takes it off each turn of a conversation, so those turns are saved.
- While Hermes runs, the entry has a provider profile of the plugin's (pricing only, hidden from pickers), and
  `@adaptiq:` is a reference prefix. Neither is written to a file.

If you had added an endpoint for the same URL by hand, the plugin takes that entry over instead of
adding a second one, and drops the key you pasted into it.

The plugin uses Hermes' own config writer (`hermes_cli.config`), which is not a documented plugin
API, so a Hermes update could move it.
