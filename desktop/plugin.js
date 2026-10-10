// AdaptiqAI Desktop half: copied by the app into ~/.hermes/desktop-plugins/adaptiq/.
// Plain ESM (no JSX, no build). Only @hermes/plugin-sdk, react and react/jsx-runtime are importable.
import { atom, Button, cn, host, ListRow, PALETTE_AREA, STATUSBAR_AREAS, Tip, useValue } from '@hermes/plugin-sdk'
import { useEffect } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const state = atom({ busy: false, signedIn: false, workspace: '', account: null, console: '' })
// Whether new messages of each conversation are saved, by Hermes session id, as the agent half keeps it.
const saving = atom({})

async function refresh(ctx) {
  try {
    const [s, a] = await Promise.all([ctx.rest('/status'), ctx.rest('/account')])
    state.set({ ...state.get(), signedIn: Boolean(s.signed_in), workspace: s.workspace || '', account: a.account, console: a.console })
    // A choice changed in the chat (/adaptiq private, /adaptiq remember) shows here too.
    for (const session of Object.keys(saving.get())) void loadSaving(ctx, session)
    return s
  } catch {
    return { signed_in: false }
  }
}

const sleep = (ctx, ms) => new Promise(resolve => ctx.setTimeout(resolve, ms))

async function signIn(ctx) {
  if (state.get().busy) return
  state.set({ ...state.get(), busy: true })
  try {
    const start = await ctx.rest('/login/start', { method: 'POST' })
    await ctx.os.openExternal(start.verification_uri_complete)
    host.notify({ kind: 'info', message: `Approve AdaptiqAI in your browser (code ${start.user_code}).` })
    const deadline = Date.now() + (start.expires_in || 600) * 1000
    while (Date.now() < deadline) {
      await sleep(ctx, (start.interval || 2) * 1000)
      const r = await ctx.rest('/login/poll', { method: 'POST', body: { device_code: start.device_code } })
      if (r.status === 'pending') continue
      if (r.status === 'done') {
        host.notify({ kind: 'success', message: r.model ? `AdaptiqAI connected: ${r.models.length} models, default ${r.model}.` : 'AdaptiqAI connected. In a chat, /adaptiq models lists the models you can get.' })
      } else {
        host.notify({ kind: 'error', message: `AdaptiqAI sign-in ${r.status}.` })
      }
      return
    }
    host.notify({ kind: 'error', message: 'AdaptiqAI sign-in timed out.' })
  } catch (error) {
    host.notifyError(error, 'AdaptiqAI sign-in failed.')
  } finally {
    state.set({ ...state.get(), busy: false })
    await refresh(ctx)
  }
}

async function loadSaving(ctx, session) {
  try {
    const r = await ctx.rest(`/conversation?session=${encodeURIComponent(session)}`)
    saving.set({ ...saving.get(), [session]: Boolean(r.saving) })
  } catch {
    // Unknown: the chip stays hidden rather than guess.
  }
}

async function setSaving(ctx, session, on) {
  try {
    const r = await ctx.rest('/conversation', { method: 'POST', body: { session, saving: on } })
    saving.set({ ...saving.get(), [session]: Boolean(r.saving) })
    host.notify({ kind: 'info', message: r.saving ? 'New messages of this conversation are saved to AdaptiqAI memory again.' : 'Private: new messages of this conversation are no longer saved. Existing memory is kept.' })
  } catch (error) {
    host.notifyError(error, 'AdaptiqAI could not change this conversation.')
  }
}

function Account({ ctx }) {
  const s = useValue(state)
  const account = s.account
  const rows = [
    jsx(ListRow, {
      title: s.signedIn ? `Connected to ${account ? account.workspace : s.workspace}` : 'Not connected',
      description: account && account.memory ? `Memory ${account.memory}` : s.signedIn ? '' : 'Sign in once in the browser to use your workspace’s models in Hermes.',
      action: jsx(Button, {
        disabled: s.busy,
        onClick: () => void signIn(ctx),
        children: s.busy ? 'Waiting for the browser…' : s.signedIn ? 'Sign in again' : 'Sign in with AdaptiqAI'
      })
    })
  ]
  if (account && account.ends) {
    rows.push(jsx(ListRow, { title: account.ends, description: 'Sign in again to renew it.' }))
  }
  for (const model of (account && account.models) || []) {
    rows.push(jsx(ListRow, { title: model.name, description: model.detail }))
  }
  if (s.signedIn && account && account.models.length === 0) {
    rows.push(jsx(ListRow, { title: 'No models yet', description: 'In a chat, /adaptiq models lists the models you can get, and /adaptiq get <number> gets one.' }))
  }
  if (s.signedIn && s.console) {
    rows.push(jsx(ListRow, {
      title: 'AdaptiqAI Knowledge',
      description: 'Your collections, conversations and memory, in the browser.',
      action: jsx(Button, { onClick: () => void ctx.os.openExternal(s.console), children: 'Open' })
    }))
  }
  return jsxs('div', { className: 'grid gap-1 text-sm', children: rows })
}

// "AdaptiqAI · saved" in the status bar for the conversation in focus; a click keeps it private.
function SavingChip({ ctx }) {
  const session = useValue(host.state.focusedStoredSessionId)
  const model = useValue(host.state.model)
  const s = useValue(state)
  const known = useValue(saving)
  // Read again whenever the conversation in focus changes: the chat may have changed it meanwhile.
  useEffect(() => {
    if (session && s.signedIn) void loadSaving(ctx, session)
  }, [session, s.signedIn])
  const onAdaptiq = Boolean(s.account && s.account.models.some(entry => entry.id === model))
  if (!session || !onAdaptiq || known[session] === undefined) return null
  const on = known[session]
  return jsx(Tip, {
    label: on
      ? 'New messages of this conversation are saved to AdaptiqAI memory. Click to keep it private.'
      : 'This conversation is private: nothing new is saved. Click to save it again.',
    children: jsx('button', {
      type: 'button',
      className: cn('inline-flex h-full items-center px-1.5 text-[0.6875rem]', 'text-(--ui-text-tertiary)'),
      onClick: () => void setSaving(ctx, session, !on),
      children: on ? 'AdaptiqAI · saved' : 'AdaptiqAI · private'
    })
  })
}

export default {
  id: 'adaptiq', // must equal the folder name the app copies this half into
  name: 'AdaptiqAI',
  register(ctx) {
    ctx.registerSettingsPage?.({
      id: 'account',
      title: 'AdaptiqAI',
      icon: 'account',
      render: function AccountPage() {
        return jsx(Account, { ctx })
      }
    })
    ctx.register({
      id: 'saving',
      area: STATUSBAR_AREAS.right,
      order: 120,
      render: () => jsx(SavingChip, { ctx })
    })
    ctx.register({
      id: 'sign-in',
      area: PALETTE_AREA,
      data: { id: 'adaptiq.sign-in', label: 'AdaptiqAI: Sign in', keywords: ['adaptiq', 'login'], run: () => void signIn(ctx) }
    })
    // The models and memory change as lanes start and stop: read them again now and then.
    ctx.setInterval(() => void refresh(ctx), 60000)
    // First load after install: start the sign-in once, so the install click is the only click.
    void refresh(ctx).then(s => {
      if (!s.signed_in && !ctx.storage.get('autoStarted', false)) {
        ctx.storage.set('autoStarted', true)
        void signIn(ctx)
      }
    })
  }
}
