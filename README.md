# AdaptiqAI for Hermes

Use your AdaptiqAI workspace's models in Hermes. You sign in once in the browser, so there is no
key to copy or paste.

## Install

- **Hermes Desktop:** open `hermes://plugin/install?repo=AdaptiqAI/hermes-plugin`, or use the
  "Install in Hermes" button in the AdaptiqAI console. Click Install. Sign-in starts on its own,
  and your browser asks you to approve the connection.
- **Terminal:** `hermes plugins install AdaptiqAI/hermes-plugin`, then `hermes adaptiq login`.

## Signing in again

- **Desktop:** Settings → AdaptiqAI → Sign in again, or "AdaptiqAI: Sign in" in the command
  palette.
- **Any chat:** `/adaptiq`, or `/adaptiq status`.
- **Terminal:** `hermes adaptiq login`, or `hermes adaptiq status`.

A connection lasts 90 days. You can see it in the console under **API keys**, as
"Hermes on <your computer>", and revoking it there disconnects this computer only.

## What it changes in Hermes

- `ADAPTIQ_API_KEY` in Hermes' `.env`.
- A `providers.adaptiq` entry pointing at `https://api.adaptiqai.com/v1`, with the key taken from
  that variable. The model list is read live from the workspace, so a model rented later appears
  without signing in again.
- The main model becomes the workspace model with the most lanes.

If you had added an endpoint for the same URL by hand, the plugin takes that entry over instead of
adding a second one, and drops the key you pasted into it.

The plugin uses Hermes' own config writer (`hermes_cli.config`), which is not a documented plugin
API, so a Hermes update could move it.
