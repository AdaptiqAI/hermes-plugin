# AdaptiqAI for Hermes

Use your AdaptiqAI workspace's models in Hermes. You sign in once in the browser, so there is no
key to copy or paste. Each Hermes conversation keeps its own native memory on AdaptiqAI.

## Install

- **Hermes Desktop:** open `hermes://plugin/install?repo=AdaptiqAI/hermes-plugin`, or use the
  "Install in Hermes" button in the AdaptiqAI console. Click Install. Sign-in starts on its own,
  and your browser asks you to approve the connection.
- **Terminal:** `hermes plugins install AdaptiqAI/hermes-plugin`, then `hermes adaptiq login`.

## Hermes Desktop

- **Settings → AdaptiqAI** shows:
  - the workspace and memory use;
  - each model with its lanes and how many run now, with the main model marked;
  - in its last two weeks, when the connection ends.

  It has buttons to sign in (again) and to open AdaptiqAI Knowledge. A new workspace is told how to get its
  first model.
- **The status bar** shows "AdaptiqAI · saved" or "AdaptiqAI · private" for the conversation in focus, while it
  runs on an AdaptiqAI model. A click switches it, like `/adaptiq private` and `/adaptiq remember`.

## Signing in again

- **Desktop:** Settings → AdaptiqAI → Sign in again, or "AdaptiqAI: Sign in" in the command
  palette.
- **Any chat:** `/adaptiq`, or `/adaptiq status`. Status shows the workspace, library and conversation-history use, each model
  with its lanes and how many run now, and, in its last two weeks, when the connection ends. Where AdaptiqAI puts idle conversations to sleep to
  make room, one line says so; in ask mode, a suggestion gives the console link.
- **Terminal:** `hermes adaptiq login`, or `hermes adaptiq status`.

A connection lasts 90 days. You can see it in the console under **API keys**, as
"Hermes on <your computer>", and revoking it there disconnects this computer only.

## Getting a model

A new workspace has no model yet, and you can get one without leaving Hermes:

- **`/adaptiq models`** lists the models on offer. For each it shows the price per lane, the speed, the region,
  and what your workspace already has.
- **`/adaptiq get <number> [lanes]`** opens a page where the workspace owner confirms the lanes, and pays by card
  if the offer asks for it (hourly lanes run from the workspace balance). Each lane runs one conversation at a
  time.

The plugin follows the purchase until the lanes are on. Then the model joins Hermes' model list. If Hermes' main
model has no lanes, the new model becomes the main model.

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
  started it, so the AdaptiqAI console lists it there, as a helper.

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

- **`adaptiq_knowledge`** (agent tool) says what the conversation reads: its collections and the conversations
  attached to it. It searches the library and attaches or detaches a collection, which applies from the next
  model call.
- **`adaptiq_memory`** (agent tool) reads the workspace's memory use and lists collections with their sizes. It
  gives a link to AdaptiqAI Knowledge for cleaning up, and stops saving the conversation when you ask it to.
  Agents never delete anything; clean-up happens in the console.
- **`adaptiq_create_knowledge`** (agent tool) makes a collection from text, files on this computer, or HTTPS
  links, which can be GitHub repositories. It does so only when you ask for it in your own words, and never
  from a subagent. Once the collection is ready, it is used in the conversation.
- **`/adaptiq knowledge`** lists the knowledge the conversation uses.

The tools are offered while AdaptiqAI is connected and is Hermes' main provider. Hermes lists plugin tools
behind its tool search, so the agent finds them by name when it needs them.

## When something goes wrong

The plugin tells Hermes what AdaptiqAI's answers mean, so Hermes recovers the right way:

- **No lanes on a model, or a lane stopped while idle.** Hermes says the model isn't available and
  that you can pick another with `/model`. It no longer tells you to fix your API key.
- **Memory full.** Hermes stops instead of retrying, and never compresses the conversation (AdaptiqAI
  keeps it whole). You can free up space in Knowledge, use another model, or continue with
  `/adaptiq private`. Where AdaptiqAI puts idle conversations to sleep to make room (their compiled
  memory is dropped, the transcript is kept, and they wake when you use them; nothing is deleted),
  a full memory means nothing idle could sleep, and Hermes says: "Memory is full and nothing idle
  could sleep. Delete or unpin a conversation in the console, or add memory." A new conversation
  frees nothing.
- **All lanes busy, or a conversation being prepared.** Hermes retries by itself.
- **Expired sign-in.** Hermes asks you to run `/adaptiq login`.

Hermes Desktop shows these in AdaptiqAI's own words.

## What it changes in Hermes

- `ADAPTIQ_API_KEY` in Hermes' `.env`.
- A `providers.AdaptiqAI` entry pointing at `https://api.adaptiqai.com/v1`, with the key taken from
  that variable. The model list is read live from the workspace, so a model rented later appears
  without signing in again.
- The main model becomes the workspace model with the most lanes.
- The entry follows your lanes. The plugin brings it in step when a conversation starts (at most every
  ten minutes) and on `/adaptiq status`:
  - A model rented later gets its context window.
  - A model whose lanes ended leaves the entry (the engine's own model stays listed without lanes, and is never made the main model).
  - If Hermes' main model was one of those, it moves to the model with the most lanes.
- `compression.model_thresholds` gets `1.0` for each AdaptiqAI model that has no threshold of its
  own, so Hermes compacts those conversations only when they no longer fit (see Memory per
  conversation). A threshold you set yourself is kept.
- The entry's `extra_body` marks requests to it as run-once (`adaptiq_native_memory`). Hermes' helper
  calls inherit that. The plugin takes it off each turn of a conversation, so those turns are saved.

If you had added an endpoint for the same URL by hand, the plugin takes that entry over instead of
adding a second one, and drops the key you pasted into it.

The plugin uses Hermes' own config writer (`hermes_cli.config`), which is not a documented plugin
API, so a Hermes update could move it.
