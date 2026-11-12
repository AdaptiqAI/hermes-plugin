// AdaptiqAI Desktop half: copied by the app into ~/.hermes/desktop-plugins/adaptiq/.
// Plain ESM (no JSX, no build). Only @hermes/plugin-sdk, react and react/jsx-runtime are importable.
import {
  atom, Button, cn, host, ListRow, PALETTE_AREA, Popover, PopoverContent, PopoverTrigger, STATUSBAR_AREAS, useValue
} from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const state = atom({ busy: false, signedIn: false, workspace: '', account: null, console: '', action: null })
// Each conversation as the status chip shows it, by Hermes session id: whether new messages are saved, as the agent
// half keeps it, how much memory holds, and the model's window.
const saving = atom({})
// The memory warning level already raised, the sleeping lines, the newer version and the refusal already said: each
// is said once.
const told = { level: 0, sleeping: new Set(), update: '', action: 0 }
// What the chip says for a few seconds after a reply read from memory (DSH's "+24.6k from memory").
const flash = atom({})
const AFTER_REPLY_MS = 4000

/** A token count, short (core's formatTokens): 950, 1.2k, 130k, 2.29M. */
function tokens(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '0'
  const trim = (number, digits) => String(Number(number.toFixed(digits)))
  if (value < 1000) return String(Math.round(value))
  if (value < 1000000) return trim(value / 1000, value < 100000 ? 1 : 0) + 'k'
  return trim(value / 1000000, value < 10000000 ? 2 : 1) + 'M'
}

/**
 * The memory warning, sleeping conversations and a newer plugin, said once each, with the way to the console
 * (DSH's notices). Hermes can't update a plugin by itself, so the update says the command that does.
 */
function notices(ctx, account, consoleUrl, update, action) {
  const warning = account && account.warning
  const level = warning ? warning.level : 0
  if (level > told.level && warning) {
    host.notify({
      id: 'adaptiq-memory', kind: level === 2 ? 'error' : 'warning', title: 'AdaptiqAI memory', message: warning.text,
      action: { label: 'Open console', onClick: () => void ctx.os.openExternal(consoleUrl) }
    })
  }
  told.level = level
  for (const line of (account && account.sleeping) || []) {
    if (told.sleeping.has(line)) continue
    told.sleeping.add(line)
    host.notify({ kind: 'info', message: line })
  }
  if (update && update.version !== told.update) {
    told.update = update.version
    host.notify({ id: 'adaptiq-update', kind: 'info', title: 'AdaptiqAI update', message: update.text })
  }
  // A refusal the person can end: sign in again, add funds or get access, with the button that does it.
  if (action && action.at !== told.action) {
    told.action = action.at
    host.notify({
      id: 'adaptiq-action', kind: 'error', title: action.title, message: action.text,
      action: { label: action.label, onClick: () => runAction(ctx, action) }
    })
  }
}

function runAction(ctx, action) {
  if (action.kind === 'sign_in') void signIn(ctx)
  else if (action.url) void ctx.os.openExternal(action.url)
}

async function refresh(ctx) {
  try {
    const [s, a] = await Promise.all([ctx.rest('/status'), ctx.rest('/account')])
    state.set({ ...state.get(), signedIn: Boolean(s.signed_in), workspace: s.workspace || '', account: a.account, console: a.console, action: a.action || null })
    notices(ctx, a.account, a.console, a.update, a.action)
    // A choice changed in the chat (/adaptiq private, /adaptiq remember) shows here too.
    for (const session of Object.keys(saving.get())) void loadSaving(ctx, session, (saving.get()[session] || {}).model)
    return s
  } catch {
    return { signed_in: false }
  }
}

async function signOut(ctx) {
  try {
    const r = await ctx.rest('/logout', { method: 'POST' })
    host.notify({ kind: 'info', message: r.signed_in ? 'Signed out of AdaptiqAI on this computer. An API key is still set up, so Hermes stays connected.' : 'Signed out of AdaptiqAI on this computer. Your saved memory stays in your workspace.' })
  } catch (error) {
    host.notifyError(error, 'AdaptiqAI could not sign out.')
  } finally {
    await refresh(ctx)
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
        host.notify({ kind: 'success', message: r.model ? `AdaptiqAI connected: ${r.models.length} models, default ${r.model}.` : 'AdaptiqAI connected. Your workspace has no models yet: open the AdaptiqAI console to see them.' })
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

async function loadSaving(ctx, session, model) {
  try {
    const r = await ctx.rest(`/conversation?session=${encodeURIComponent(session)}&model=${encodeURIComponent(model || '')}`)
    const before = saving.get()[session]
    const processed = r.processed || null
    saving.set({ ...saving.get(), [session]: { saving: Boolean(r.saving), held: r.memory_tokens, window: r.window, model, processed, free: Boolean(r.memory_read_free) } })
    // A reply counted since the last read, that read from memory: the chip says so for a moment. The count already
    // there when the conversation first shows is history, not news.
    const turn = processed && processed.turn
    if (before && before.processed !== undefined && processed && processed.at !== (before.processed && before.processed.at) && turn && turn.from_memory > 0) {
      flash.set({ session, fromMemory: turn.from_memory, free: Boolean(r.memory_read_free) })
      ctx.setTimeout(() => { if (flash.get().session === session) flash.set({}) }, AFTER_REPLY_MS)
    }
  } catch {
    // Unknown: the chip stays hidden rather than guess.
  }
}

async function setSaving(ctx, session, on) {
  try {
    const r = await ctx.rest('/conversation', { method: 'POST', body: { session, saving: on } })
    saving.set({ ...saving.get(), [session]: { ...(saving.get()[session] || {}), saving: Boolean(r.saving) } })
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
      title: s.signedIn ? `Connected to ${account ? account.workspace : s.workspace}${account && account.api_key ? ' (API key)' : ''}` : 'Not connected',
      description: s.signedIn ? '' : 'Sign in once in the browser to use your workspace’s models in Hermes.',
      action: jsx(Button, {
        disabled: s.busy,
        onClick: () => void signIn(ctx),
        children: s.busy ? 'Waiting for the browser…' : s.signedIn ? 'Sign in again' : 'Sign in with AdaptiqAI'
      })
    })
  ]
  if (account && account.memory) {
    rows.push(jsx(ListRow, { title: `Memory: ${account.memory}`, description: account.split || '' }))
  }
  if (account && account.warning) {
    rows.push(jsx(ListRow, { title: account.warning.text }))
  }
  if (account && account.ends) {
    rows.push(jsx(ListRow, { title: account.ends, description: 'Sign in again to renew it.' }))
  }
  for (const model of (account && account.models) || []) {
    rows.push(jsx(ListRow, { title: model.name, description: model.detail }))
  }
  if (s.signedIn && account && account.models.length === 0) {
    rows.push(jsx(ListRow, { title: 'No models yet', description: 'Open the AdaptiqAI console to see your workspace’s models.' }))
  }
  if (s.signedIn && s.console) {
    rows.push(jsx(ListRow, {
      title: 'AdaptiqAI Knowledge',
      description: 'Your collections, conversations and memory, in the browser.',
      action: jsx(Button, { onClick: () => void ctx.os.openExternal(s.console), children: 'Open' })
    }))
  }
  if (s.signedIn) {
    rows.push(jsx(ListRow, {
      title: 'Sign out',
      description: 'Forget this computer’s sign-in. Your saved memory stays in your workspace; an API key set up for Hermes stays.',
      action: jsx(Button, { disabled: s.busy, onClick: () => void signOut(ctx), children: 'Sign out' })
    }))
  }
  return jsxs('div', { className: 'grid gap-1 text-sm', children: rows })
}

/** Dollars with the cents a single turn needs (core's formatUsd): $1.24, $0.24, $0.0046, <$0.0001. */
function usd(value) {
  const amount = typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0
  if (amount > 0 && amount < 0.0001) return '<' + usd(0.0001)
  const digits = amount === 0 || amount >= 0.1 ? 2 : 4
  try {
    return new Intl.NumberFormat('en', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: digits }).format(amount)
  } catch {
    return '$' + amount.toFixed(digits)
  }
}

/** A share, as core says it: <1%, 96%. */
function percent(fraction) {
  if (!Number.isFinite(fraction) || fraction <= 0) return '0%'
  if (fraction < 0.01) return '<1%'
  return Math.min(100, Math.round(fraction * 100)) + '%'
}

const heading = text => jsx('div', { className: 'pt-2 text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)', children: text })
const row = (label, ...values) => jsxs('div', {
  className: 'flex items-center justify-between gap-3',
  children: [jsx('span', { className: 'text-(--ui-text-secondary)', children: label }), ...values.map(value => jsx('span', { className: 'tabular-nums', children: value }))]
})

/** Core's Processed table: the last turn and the whole conversation, with why it outgrows the memory. */
function Processed({ processed, free }) {
  const turn = processed.turn
  const all = processed.conversation
  const steps = view => view.requests === 1 ? '1 step' : `${view.requests.toLocaleString('en')} steps`
  const cost = view => view.cost_usd === null || view.cost_usd === undefined ? '—' : usd(view.cost_usd)
  const read = all.new_input + all.from_memory + all.written
  const share = read > 0 ? all.from_memory / read : null
  const rows = [
    heading('Processed'),
    row('', 'Last turn', 'This conversation'),
    row('Steps', steps(turn), steps(all)),
    row('New input', tokens(turn.new_input), tokens(all.new_input)),
    row('Read from memory', tokens(turn.from_memory), tokens(all.from_memory)),
    row('Saved to memory', tokens(turn.written), tokens(all.written)),
    row('Output', tokens(turn.output), tokens(all.output)),
    row('Cost', cost(turn), cost(all))
  ]
  if (share !== null && all.from_memory > 0) {
    rows.push(jsx('p', {
      className: 'text-(--ui-text-tertiary)',
      children: `Each step reads the whole memory again. ${percent(share)} came from memory${free ? ', which is free' : ''}.`
    }))
  }
  if (processed.since) {
    rows.push(jsx('p', { className: 'text-(--ui-text-tertiary)', children: `Counted on this computer since ${new Date(processed.since * 1000).toLocaleDateString()}` }))
  }
  return jsxs('div', { className: 'grid gap-1', children: rows })
}

/** The pill's panel (DSH's popover): saving, the knowledge in use with more to use in one click, what was processed,
 *  and the button a refusal asks for. */
function MemoryPanel({ ctx, session, model, chip }) {
  const s = useValue(state)
  const [known, setKnown] = useState(null)
  const [busy, setBusy] = useState('')
  const query = `session=${encodeURIComponent(session)}&model=${encodeURIComponent(model || '')}`
  useEffect(() => {
    let live = true
    ctx.rest(`/knowledge?${query}`)
      .then(r => { if (live) setKnown(r) })
      .catch(() => { if (live) setKnown({ error: true }) })
    return () => { live = false }
  }, [query])
  async function use(memoryId, on) {
    setBusy(memoryId)
    try {
      setKnown(await ctx.rest('/knowledge', { method: 'POST', body: { session, model, memory_id: memoryId, use: on } }))
    } catch (error) {
      host.notifyError(error, 'AdaptiqAI could not change this conversation’s knowledge.')
    } finally {
      setBusy('')
    }
  }
  const children = [
    jsx('div', { className: 'font-medium', children: 'Memory' }),
    typeof chip.held === 'number' && chip.held > 0
      ? jsx('p', { className: 'text-(--ui-text-secondary)', children: `This conversation holds ${tokens(chip.held)}${typeof chip.window === 'number' ? ` of ${tokens(chip.window)}` : ''} in AdaptiqAI memory.` })
      : null,
    jsxs('div', {
      className: 'flex items-center justify-between gap-2 pt-1',
      children: [
        jsx('span', { children: 'Saving' }),
        jsxs('div', {
          className: 'flex gap-1',
          children: [
            jsx(Button, { size: 'sm', variant: chip.saving ? 'default' : 'outline', onClick: () => void setSaving(ctx, session, true), children: 'Remember' }),
            jsx(Button, { size: 'sm', variant: chip.saving ? 'outline' : 'default', onClick: () => void setSaving(ctx, session, false), children: 'Private' })
          ]
        })
      ]
    }),
    heading('Knowledge')
  ]
  if (known === null) children.push(jsx('p', { className: 'text-(--ui-text-tertiary)', children: 'Reading…' }))
  else if (known.error) children.push(jsx('p', { className: 'text-(--ui-text-tertiary)', children: 'AdaptiqAI could not read this conversation’s knowledge.' }))
  else {
    for (const item of known.used) {
      children.push(row(item.name, jsx(Button, { size: 'sm', variant: 'ghost', disabled: busy !== '', title: 'Click to stop using it here. Nothing is deleted.', onClick: () => void use(item.memory_id, false), children: busy === item.memory_id ? '…' : 'In use' })))
    }
    for (const item of known.others) {
      children.push(row(item.name, jsx(Button, { size: 'sm', variant: 'outline', disabled: busy !== '', onClick: () => void use(item.memory_id, true), children: busy === item.memory_id ? '…' : 'Use here' })))
    }
    if (known.used.length === 0 && known.others.length === 0) {
      children.push(jsx('p', { className: 'text-(--ui-text-tertiary)', children: 'No collection is ready for this model yet.' }))
    }
    children.push(jsx(Button, { size: 'sm', variant: 'ghost', onClick: () => void ctx.os.openExternal(known.console), children: 'Open in Knowledge' }))
  }
  if (chip.processed && chip.processed.conversation) children.push(jsx(Processed, { processed: chip.processed, free: chip.free }))
  if (s.action) {
    children.push(jsxs('div', {
      className: 'grid gap-1 pt-2',
      children: [
        jsx('p', { children: s.action.text }),
        jsx(Button, { size: 'sm', onClick: () => runAction(ctx, s.action), children: s.action.label })
      ]
    }))
  }
  return jsxs('div', { className: 'grid gap-1 text-xs', children: children.filter(Boolean) })
}

// DSH's memory pill in the status bar, for the conversation in focus: "AdaptiqAI · 25k / 5M", how much of the
// conversation memory holds against its model's window, and "private" when nothing new is saved; for a moment after a
// reply, what it read from memory. A click opens the panel: saving, knowledge, what was processed.
function MemoryChip({ ctx }) {
  const session = useValue(host.state.focusedStoredSessionId)
  const model = useValue(host.state.model)
  const usage = useValue(host.state.focusedUsage)
  const busy = useValue(host.state.busy)
  const s = useValue(state)
  const known = useValue(saving)
  const after = useValue(flash)
  const [open, setOpen] = useState(false)
  // Read again whenever the conversation in focus, its model or its usage (a reply) changes.
  useEffect(() => {
    if (session && s.signedIn) void loadSaving(ctx, session, model)
  }, [session, model, s.signedIn, usage])
  // A turn that ended may have been refused: the account says what the person can do about it.
  useEffect(() => {
    if (!busy) void refresh(ctx)
  }, [busy])
  const onAdaptiq = Boolean(s.account && s.account.models.some(entry => entry.id === model))
  const chip = session ? known[session] : undefined
  if (!session || !onAdaptiq || chip === undefined) return null
  const on = chip.saving
  const size = typeof chip.held === 'number' && chip.held > 0
    ? (typeof chip.window === 'number' ? `${tokens(chip.held)} / ${tokens(chip.window)}` : tokens(chip.held))
    : (on ? 'saved' : '')
  const label = after.session === session
    ? ['AdaptiqAI', `+${tokens(after.fromMemory)} from memory${after.free ? ' · free' : ''}`].join(' · ')
    : ['AdaptiqAI', size, on ? '' : 'private'].filter(Boolean).join(' · ')
  return jsxs(Popover, {
    open,
    onOpenChange: setOpen,
    children: [
      jsx(PopoverTrigger, {
        asChild: true,
        children: jsx('button', {
          type: 'button',
          title: (typeof chip.held === 'number' && chip.held > 0 ? `This conversation holds ${tokens(chip.held)} tokens in AdaptiqAI memory. ` : '')
            + (on ? 'New messages are saved to AdaptiqAI memory.' : 'This conversation is private: nothing new is saved.'),
          className: cn('inline-flex h-full items-center px-1.5 text-[0.6875rem]', 'text-(--ui-text-tertiary)'),
          children: label
        })
      }),
      jsx(PopoverContent, {
        side: 'top',
        align: 'end',
        className: 'w-80',
        children: open ? jsx(MemoryPanel, { ctx, session, model, chip }) : null
      })
    ]
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
      render: () => jsx(MemoryChip, { ctx })
    })
    ctx.register({
      id: 'sign-in',
      area: PALETTE_AREA,
      data: { id: 'adaptiq.sign-in', label: 'AdaptiqAI: Sign in', keywords: ['adaptiq', 'login', 'signin'], run: () => void signIn(ctx) }
    })
    ctx.register({
      id: 'sign-out',
      area: PALETTE_AREA,
      data: { id: 'adaptiq.sign-out', label: 'AdaptiqAI: Sign out', keywords: ['adaptiq', 'logout', 'signout'], run: () => void signOut(ctx) }
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
