class UIRequestError extends Error {
    constructor(message, auth = false) { super(message); this.auth = auth; }
}
let requestNoticeOwner = null;
let requestNoticeKey = null;
let requestNoticePreviousFocus = null;
const dismissedRequestNotices = new Map();
function showRequestNotice(message, {auth = false, retry = null, owner = null} = {}) {
    const key = JSON.stringify([message, auth]);
    if (dismissedRequestNotices.get(owner) === key) return;
    const notice = document.getElementById('requestNotice');
    if (!notice.hidden && requestNoticeOwner === owner && requestNoticeKey === key) return;
    if (notice.hidden && document.activeElement && !notice.contains(document.activeElement)) {
        requestNoticePreviousFocus = document.activeElement;
    }
    requestNoticeOwner = owner;
    requestNoticeKey = key;
    document.getElementById('requestNoticeText').textContent = message;
    const signIn = document.getElementById('requestSignIn');
    signIn.hidden = !auth;
    signIn.href = '/login?next=' + encodeURIComponent(window.location.pathname + window.location.search);
    const button = document.getElementById('requestRetry');
    button.hidden = !retry || auth;
    button.onclick = retry;
    notice.hidden = false;
}
function dismissRequestNotice() {
    dismissedRequestNotices.set(requestNoticeOwner, requestNoticeKey);
    document.getElementById('requestNotice').hidden = true;
    if (requestNoticePreviousFocus?.isConnected) requestNoticePreviousFocus.focus();
}
function clearRequestNotice(owner = null) {
    dismissedRequestNotices.delete(owner);
    if (requestNoticeOwner === owner) {
        document.getElementById('requestNotice').hidden = true;
        requestNoticeKey = null;
    }
}
function reportRequestError(error, message, retry = null, owner = null) {
    const auth = error instanceof UIRequestError && error.auth;
    showRequestNotice(auth ? 'Your session has ended. Sign in again to continue.' : message, {auth, retry, owner});
}
async function requestJSON(url, options = {}) {
    let response;
    try { response = await fetch(url, {credentials: 'include', ...options}); }
    catch (_) { throw new UIRequestError('The request could not be completed.'); }
    const loginRedirect = response.redirected && new URL(response.url, window.location.origin).pathname === '/login';
    if (response.status === 401 || loginRedirect) throw new UIRequestError('Sign in required.', true);
    if (!response.ok) throw new UIRequestError('The request could not be completed.');
    try { return await response.json(); }
    catch (_) { throw new UIRequestError('The request could not be completed.'); }
}
async function runUIAction(buttons, work, message) {
    const controls = (Array.isArray(buttons) ? buttons : [buttons]).filter(Boolean);
    if (controls.some(button => button.disabled)) return;
    controls.forEach(button => { button.disabled = true; });
    try { await work(); clearRequestNotice(); }
    catch (error) { reportRequestError(error, message); }
    finally { controls.forEach(button => { button.disabled = false; }); }
}
