// AdaptiqAI Desktop half: copied by the app into ~/.hermes/desktop-plugins/adaptiq/.
// Plain ESM (no JSX, no build). Only @hermes/plugin-sdk and react/jsx-runtime are importable.
import { atom, Button, host, PALETTE_AREA, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

const state = atom({ busy: false, signedIn: false, workspace: '' })

async function refresh(ctx) {
  try {
    const s = await ctx.rest('/status')
    state.set({ ...state.get(), signedIn: Boolean(s.signed_in), workspace: s.workspace || '' })
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
        host.notify({ kind: 'success', message: `AdaptiqAI connected: ${r.models.length} models, default ${r.model}.` })
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

export default {
  id: 'adaptiq', // must equal the folder name the app copies this half into
  name: 'AdaptiqAI',
  register(ctx) {
    ctx.registerSettingsPage?.({
      id: 'account',
      title: 'AdaptiqAI',
      icon: 'account',
      render: function Account() {
        const s = useValue(state)
        return jsxs('div', {
          className: 'grid gap-3 text-sm',
          children: [
            jsx('p', { children: s.signedIn ? `Connected to ${s.workspace}.` : 'Not connected.' }),
            jsx(Button, {
              disabled: s.busy,
              onClick: () => void signIn(ctx),
              children: s.busy ? 'Waiting for the browser…' : s.signedIn ? 'Sign in again' : 'Sign in with AdaptiqAI'
            })
          ]
        })
      }
    })
    ctx.register({
      id: 'sign-in',
      area: PALETTE_AREA,
      data: { id: 'adaptiq.sign-in', label: 'AdaptiqAI: Sign in', keywords: ['adaptiq', 'login'], run: () => void signIn(ctx) }
    })
    // First load after install: start the sign-in once, so the install click is the only click.
    void refresh(ctx).then(s => {
      if (!s.signed_in && !ctx.storage.get('autoStarted', false)) {
        ctx.storage.set('autoStarted', true)
        void signIn(ctx)
      }
    })
  }
}
