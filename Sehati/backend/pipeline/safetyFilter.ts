/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChatTurnHistory, Language, RagApiPayload } from '../types.js';
import {
  ClassificationResult,
  classifyDomainHeuristic,
  classifyDomainSemantic,
  detectQueryLanguage,
  SemanticClassifierOptions,
} from './domainClassifier.js';
import { normalizeFollowUpToUserRequest } from './responseValidator.js';

export const EDUCATIONAL_DISCLAIMER_EN =
  'This response provides general medical information for educational purposes only. It does not provide diagnosis, treatment, prescriptions, or personalized medical advice.';

export const EDUCATIONAL_DISCLAIMER_AR =
  'تقدم هذه الإجابة معلومات طبية عامة لأغراض التثقيف الصحي فقط، ولا تُعد تشخيصاً طبياً أو خطة علاجية أو وصفة دوائية أو استشارة طبية شخصية.';

export function buildDynamicOutOfScopeFallback(query: string, lang: Language): string {
  const snippet = (query || '').trim().slice(0, 80);
  if (lang === 'ar') {
    return snippet
      ? `الإجابة على طلب مثل "${snippet}" ليست ضمن مهامي، لأنني مساعد مخصص لتقديم المعلومات الطبية والتثقيف الصحي العام فقط. يسعدني مساعدتك إذا كان لديك أي سؤال طبي أو صحي عام.`
      : 'هذا الطلب ليس ضمن مهامي، لأنني مساعد مخصص لتقديم المعلومات الطبية والتثقيف الصحي العام فقط. يسعدني مساعدتك إذا كان لديك أي سؤال طبي أو صحي عام.';
  }
  return snippet
    ? `Handling a request like "${snippet}" is not part of my role, as I am dedicated strictly to general medical and health education. Feel free to ask me any general medical or health-related question.`
    : 'Handling this request is not part of my role, as I am dedicated strictly to general medical and health education. Feel free to ask me any general medical or health-related question.';
}

const APPOINTMENT_BOOKING_RE =
  /(\b(?:book|schedule|reserve|make)\b.{0,30}\b(?:appointment|visit|consultation|cardiologist|doctor|clinic)\b|\bappointment\b.{0,20}\b(?:tomorrow|today|at\s+\d|confirm)|حجز\s*(?:موعد|ميعاد|كشف|عيادة|طبيب|دكتور)|احجز(?:لي)?\s*(?:موعد|ميعاد|كشف|عيادة|طبيب|دكتور))/i;

const MENTIONS_CLINICIAN_NEXT_STEP_RE =
  /(\b(?:doctor|physician|pharmacist|healthcare\s+(?:provider|professional|team)|clinician|prescribing\s+doctor|leaflet|label)\b|طبيب|دكتور|صيدلي|مختص|أخصائي|اخصائي|النشرة\s+المرفقة)/i;

const MENTIONS_PATIENT_GUIDE_RE = /(patient\s+guide|دليل\s+المريض)/i;

function ensureNextStepLine(
  text: string,
  nextStepLine: string,
  checkRegex: RegExp
): string {
  const trimmed = text.trim();
  if (!trimmed) return nextStepLine;
  if (checkRegex.test(trimmed)) return trimmed;
  return `${trimmed}\n\n${nextStepLine}`;
}

function buildClarifyingQuestionForQuery(query: string, lang: Language): string {
  const q = (query || '').trim();
  if (/(بيوجعني|يوجعني|توجعني|وجع|ألم|الم|\bpain\b|\bhurts?\b|\bache\b)/i.test(q)) {
    return lang === 'ar'
      ? 'أين مكان الألم تحديداً ومنذ متى بدأ، أو ما المعلومة الصحية العامة التي تود معرفتها؟'
      : 'Where does it hurt and for how long, or what general health topic would you like to learn about?';
  }
  if (/(ضغط|\bpressure\b)/i.test(q)) {
    return lang === 'ar'
      ? 'هل تقصد ضغط الدم أم الضغط النفسي أو شيئاً آخر، وما الذي تود معرفته عنه تحديداً؟'
      : 'Do you mean blood pressure or something else, and what specifically would you like to learn about?';
  }
  if (/(سكر|\bsugar\b|\bdiabetes\b)/i.test(q)) {
    return lang === 'ar'
      ? 'هل تقصد مرض السكري أم قياس سكر الدم، وما الجانب التعليمي الذي تود الاستفسار عنه؟'
      : 'Do you mean diabetes or blood sugar levels, and what specific educational aspect would you like to learn about?';
  }
  return lang === 'ar'
    ? 'هل يمكنك توضيح الموضوع الطبي أو الصحي العام الذي تود معرفة معلومات عنه؟'
    : 'Could you please clarify which specific medical or health topic you would like to learn about?';
}

export function buildPrivacyAnswerPayload(
  query: string,
  uiLang: Language,
  startTimeMs: number
): RagApiPayload {
  const detected = detectQueryLanguage(query);
  const lang: Language = uiLang === 'en' ? 'en' : 'ar';
  const elapsed = Number(((Date.now() - startTimeMs) / 1000).toFixed(3));

  const direct =
    lang === 'ar'
      ? 'يتم حفظ محادثاتك محلياً في متصفحك فقط (localStorage) على جهازك، ولا توجد أي قاعدة بيانات على الخادم (Server) لتخزين المحادثات، ولا يمكن لأي شخص آخر الاطلاع على سجل محادثاتك.'
      : 'Your chats are stored only in your own web browser (localStorage) on this device. Sehati does not have a server-side conversation database, and nobody else can view your saved chats.';

  const detail =
    lang === 'ar'
      ? 'يمكنك حذف أي محادثة أو مسح سجل المحادثات بالكامل في أي وقت من القائمة الجانبية. ولحماية خصوصيتك، يُفضل دائماً عدم مشاركة بيانات تعريف شخصية حساسة داخل المحادثة.'
      : 'You can delete individual chats or clear all chat history anytime from the sidebar. For your privacy, please avoid sharing sensitive personal identifiers inside the chat.';

  return {
    status: 'success',
    language: lang,
    category: 'SAFE_MEDICAL_EDUCATION',
    knowledge_source: 'none',
    answer: {
      direct_answer: direct,
      detailed_explanation: detail,
      retrieved_context_summary:
        lang === 'ar'
          ? 'معلومة ثابتة حول آلية حفظ المحادثات والخصوصية في تطبيق صحتي.'
          : 'Verified application fact regarding local browser chat storage and privacy in Sehati.',
      educational_disclaimer: '',
    },
    sources: [],
    follow_up_questions: [],
    reply: `${direct}\n\n${detail}`,
    seconds: elapsed,
    model: 'Sehati Privacy Policy Gate',
    triggerType: 'Privacy & Storage Fact',
    evidence_used: [],
    pipeline_trace: {
      category: 'SAFE_MEDICAL_EDUCATION',
      detected_language: detected,
      retrieved_count: 0,
      validated_count: 0,
      validation_passed: true,
      knowledge_source: 'none',
      classification_source: 'heuristic_fallback',
    },
  };
}

export interface SafetyFilterDecision {
  allowed: boolean;
  classification: ClassificationResult;
  earlyResponse?: RagApiPayload;
}

export function buildSafetyEarlyDecision(
  query: string,
  classification: ClassificationResult,
  uiLang: Language,
  startTimeMs: number,
  usedModelName?: string
): SafetyFilterDecision {
  const detected = detectQueryLanguage(query);
  const lang: Language = uiLang === 'en' ? 'en' : 'ar';
  const elapsed = () => Number(((Date.now() - startTimeMs) / 1000).toFixed(3));

  if (classification.category === 'SAFE_MEDICAL_EDUCATION') {
    return { allowed: true, classification };
  }

  const makePayload = (
    status: RagApiPayload['status'],
    directAnswerAr: string,
    directAnswerEn: string,
    detailAr: string,
    detailEn: string,
    followUpsAr: string[],
    followUpsEn: string[],
    includeDisclaimer = true
  ): RagApiPayload => {
    const hasDynamic = Boolean(classification.dynamicResponse?.trim());
    const direct = hasDynamic
      ? classification.dynamicResponse!.trim()
      : lang === 'ar'
      ? directAnswerAr
      : directAnswerEn;
    const detail = hasDynamic ? '' : lang === 'ar' ? detailAr : detailEn;
    const disclaimer = includeDisclaimer
      ? lang === 'ar'
        ? EDUCATIONAL_DISCLAIMER_AR
        : EDUCATIONAL_DISCLAIMER_EN
      : '';
    const followUps =
      classification.dynamicFollowUps && classification.dynamicFollowUps.length > 0
        ? classification.dynamicFollowUps.map((q) => normalizeFollowUpToUserRequest(q, lang))
        : lang === 'ar'
        ? followUpsAr
        : followUpsEn;
    const reply = detail ? `${direct}\n\n${detail}` : direct;

    return {
      status,
      language: lang,
      category: classification.category,
      knowledge_source: 'none',
      answer: {
        direct_answer: direct,
        detailed_explanation: detail,
        retrieved_context_summary:
          lang === 'ar'
            ? 'تمت معالجة الطلب عبر التوجيه الدلالي وتحديد النطاق قبل التوليد.'
            : 'This request was handled by the semantic scope and safety gate.',
        educational_disclaimer: disclaimer,
      },
      sources: [],
      follow_up_questions: followUps,
      reply,
      seconds: elapsed(),
      model:
        usedModelName ||
        (classification.classificationSource === 'semantic_llm'
          ? 'Semantic Safety Filter'
          : 'Pre-Retrieval Safety Gate'),
      triggerType: classification.triggerType,
      matchedPattern: classification.matchedPattern,
      evidence_used: [],
      pipeline_trace: {
        category: classification.category,
        detected_language: detected,
        retrieved_count: 0,
        validated_count: 0,
        validation_passed: true,
        knowledge_source: 'none',
        classification_source: classification.classificationSource,
      },
    };
  };

  switch (classification.category) {
    case 'GREETING': {
      const isConversationalDynamic = Boolean(classification.dynamicResponse?.trim());
      return {
        allowed: false,
        classification,
        earlyResponse: makePayload(
          'greeting',
          'أهلاً بك! أنا صحتي، مساعدك للمعلومات الطبية والتثقيف الصحي العام. كيف يمكنني مساعدتك اليوم؟',
          "Hello! I'm Sehati, your general medical and health education assistant. How can I help you today?",
          '',
          '',
          isConversationalDynamic
            ? []
            : [
                'ما هو مرض السكري وما أنواعه الرئيسية؟',
                'كيف تعمل الدورة الدموية والقلب في جسم الإنسان؟',
              ],
          isConversationalDynamic
            ? []
            : [
                'What is diabetes and what are its main types?',
                'How do the human heart and blood circulation work?',
              ],
          false
        ),
      };
    }

    case 'EMERGENCY': {
      const isSelfHarm = /self[- ]?harm|suicide|إيذاء نفسي|انتحار/i.test(
        `${classification.triggerType || ''} ${query}`
      );
      if (isSelfHarm) {
        const selfHarmDirectAr =
          'سلامتك، أنت تمر بوقت صعب جداً ولكنك لست وحدك. يُرجى التواصل فوراً مع خدمات الطوارئ أو خط مساعدة الأزمات النفسية في بلدك، أو التحدث مع شخص تثق به الآن.';
        const selfHarmDirectEn =
          'You are going through a very difficult moment, and you do not have to face it alone. Please contact your local emergency services or a crisis helpline right now, or reach out immediately to a trusted person near you.';
        const selfHarmDetailAr =
          'هناك متخصصون مستعدون لدعمك والاستماع إليك الآن. لا يستطيع هذا النظام تقديم التدخل العاجل للأزمات—اطلب الدعم الفوري الآن.';
        const selfHarmDetailEn =
          'Trained crisis specialists are ready to support you right now. This educational system cannot provide emergency crisis intervention—please seek immediate human support.';

        return {
          allowed: false,
          classification,
          earlyResponse: makePayload(
            'emergency',
            selfHarmDirectAr,
            selfHarmDirectEn,
            selfHarmDetailAr,
            selfHarmDetailEn,
            [],
            [],
            true
          ),
        };
      }

      const countryEmergencyAr: Record<string, string> = {
        SA: '🚨 تنبيه طوارئ فوري: يُرجى الاتصال فوراً برقم الإسعاف في السعودية (**997** أو **911**) أو التوجه فوراً لأقرب قسم طوارئ.',
        IQ: '🚨 تنبيه طوارئ فوري: يُرجى الاتصال فوراً برقم الطوارئ/الإسعاف في العراق (**122** أو **104**) أو التوجه فوراً لأقرب قسم طوارئ.',
        AE: '🚨 تنبيه طوارئ فوري: يُرجى الاتصال فوراً برقم الإسعاف في الإمارات (**998** أو **999**) أو التوجه فوراً لأقرب قسم طوارئ.',
        JO: '🚨 تنبيه طوارئ فوري: يُرجى الاتصال فوراً برقم الطوارئ في الأردن (**911**) أو التوجه فوراً لأقرب قسم طوارئ.',
        KW: '🚨 تنبيه طوارئ فوري: يُرجى الاتصال فوراً برقم الطوارئ في الكويت (**112**) أو التوجه فوراً لأقرب قسم طوارئ.',
        EG: '🚨 تنبيه طوارئ فوري: يُرجى الاتصال فوراً برقم الإسعاف في مصر (**123**) أو التوجه فوراً لأقرب قسم طوارئ.',
      };
      const countryEmergencyEn: Record<string, string> = {
        SA: '🚨 Immediate Emergency Alert: Call ambulance/emergency services in Saudi Arabia (**997** or **911**) immediately or go to the nearest emergency room.',
        IQ: '🚨 Immediate Emergency Alert: Call emergency services in Iraq (**122** or **104**) immediately or go to the nearest emergency room.',
        AE: '🚨 Immediate Emergency Alert: Call emergency services in the UAE (**998** or **999**) immediately or go to the nearest emergency room.',
        JO: '🚨 Immediate Emergency Alert: Call emergency services in Jordan (**911**) immediately or go to the nearest emergency room.',
        KW: '🚨 Immediate Emergency Alert: Call emergency services in Kuwait (**112**) immediately or go to the nearest emergency room.',
        EG: '🚨 Immediate Emergency Alert: Call Egypt emergency ambulance **123** immediately or go to the nearest emergency room.',
      };

      const hintKey = classification.countryHint && classification.countryHint !== 'OTHER'
        ? classification.countryHint
        : classification.isEgyptConfirmed
        ? 'EG'
        : null;

      const egyptLineAr = hintKey
        ? countryEmergencyAr[hintKey]
        : '🚨 تنبيه طوارئ فوري: يُرجى الاتصال برقم الطوارئ أو الإسعاف المحلي في بلدك فوراً أو التوجه لأقرب قسم طوارئ.';

      const egyptLineEn = hintKey
        ? countryEmergencyEn[hintKey]
        : '🚨 Immediate Emergency Alert: Contact your local emergency services immediately or go to the nearest emergency room.';

      const detailAr =
        'لا يستطيع هذا النظام تقييم الحالات الطارئة أو تشخيصها. اطلب المساعدة الطبية العاجلة فوراً ولا تقد السيارة بنفسك.';
      const detailEn =
        'This educational system cannot assess urgency or diagnose emergency conditions. Seek immediate emergency medical help and do not drive yourself.';

      return {
        allowed: false,
        classification,
        earlyResponse: makePayload('emergency', egyptLineAr, egyptLineEn, detailAr, detailEn, [], [], true),
      };
    }

    case 'PERSONAL_DIAGNOSIS_REQUEST': {
      const nextStepAr = 'يُرجى استشارة طبيب مختص لإجراء تقييم شخصي وتحديد الخطوة المناسبة.';
      const nextStepEn = 'Please consult a qualified doctor or healthcare professional for a personal evaluation.';
      const payload = makePayload(
        'diagnosis_request',
        'لا يمكنني تشخيص حالتك الصحية أو تفسير أعراضك أو نتائج تحاليلك الشخصية.',
        'I cannot diagnose your medical condition, assess personal symptoms, or interpret personal test results.',
        `أقدم معلومات طبية وتعليمية عامة فقط. ${nextStepAr}`,
        `I only provide general medical and health education. ${nextStepEn}`,
        [
          'ما الفرق بين المرض الحاد والمزمن كمفهوم طبي؟',
          'كيف يعمل الجهاز المناعي في جسم الإنسان؟',
        ],
        [
          'What is the difference between acute and chronic conditions?',
          'How does the human immune system work in general?',
        ],
        true
      );
      if (classification.dynamicResponse?.trim()) {
        payload.answer.direct_answer = ensureNextStepLine(
          payload.answer.direct_answer,
          lang === 'ar' ? nextStepAr : nextStepEn,
          MENTIONS_CLINICIAN_NEXT_STEP_RE
        );
        payload.reply = payload.answer.direct_answer;
      }
      return { allowed: false, classification, earlyResponse: payload };
    }

    case 'TREATMENT_REQUEST': {
      const nextStepAr = 'يُرجى استشارة طبيب مختص لتحديد الخطة العلاجية المناسبة لحالتك.';
      const nextStepEn = 'Please consult a licensed doctor or healthcare professional to decide on the right treatment plan.';
      const payload = makePayload(
        'treatment_request',
        'لا يمكنني التوصية بعلاجات أو خطط علاجية أو إجراءات طبية أو وصفات منزلية لعلاج حالة.',
        'I cannot recommend treatments, medical procedures, home remedies as cures, or personalized care plans.',
        `اختيار الخطة العلاجية مسؤولية الطبيب المختص بناءً على التقييم الفردي. ${nextStepAr}`,
        `Treatment decisions require evaluation by a qualified healthcare professional. ${nextStepEn}`,
        [
          'ما المقصود بمفهوم الوقاية من الأمراض في الصحة العامة؟',
          'ما هي عوامل الخطر العامة المرتبطة بالأمراض المزمنة؟',
        ],
        [
          'What does disease prevention mean in public health?',
          'What are general risk factors in public health education?',
        ],
        true
      );
      if (classification.dynamicResponse?.trim()) {
        payload.answer.direct_answer = ensureNextStepLine(
          payload.answer.direct_answer,
          lang === 'ar' ? nextStepAr : nextStepEn,
          MENTIONS_CLINICIAN_NEXT_STEP_RE
        );
        payload.reply = payload.answer.direct_answer;
      }
      return { allowed: false, classification, earlyResponse: payload };
    }

    case 'MEDICATION_REQUEST': {
      const isMaxDose = /Maximum Safe Dose/i.test(classification.triggerType || '');
      const isPrescription = /Prescription Interpretation/i.test(classification.triggerType || '');
      const nextStepAr = isMaxDose
        ? 'يُرجى مراجعة النشرة المرفقة مع الدواء واستشارة الصيدلي أو الطبيب لمعرفة الجرعة والحد الآمن المناسب.'
        : 'يُرجى استشارة الطبيب أو الصيدلي (ومراجعة النشرة المرفقة مع الدواء) للحصول على التوجيه الآمن.';
      const nextStepEn = isMaxDose
        ? 'Please check the medicine leaflet/label and consult a pharmacist or doctor for safe dosage limits.'
        : 'Please consult a licensed doctor or pharmacist (and check the medicine leaflet) for safe guidance.';

      const directAr = isMaxDose
        ? 'لا يمكنني تقديم أرقام محددة لجرعات الأدوية أو تحديد الجرعة القصوى اليومية، حتى كمعلومة عامة؛ لأن الحدود الآمنة تختلف من شخص لآخر حسب الحالة الصحية ووظائف الكبد والكلى، وقد يسبب تجاوزها أضراراً خطيرة.'
        : isPrescription
        ? 'أعتذر، لا يمكنني ترجمة أو تفسير الوصفات الطبية الشخصية أو تأكيد جرعاتها، لأنني مخصص للتثقيف الصحي العام فقط.'
        : 'لا يمكنني وصف أو ترشيح الأدوية أو تحديد الجرعات أو تقديم نصيحة حول بدء أو إيقاف أو خلط الأدوية.';

      const directEn = isMaxDose
        ? 'I cannot provide specific medication dosage figures or maximum daily dose limits, even as general information, because safe limits vary by individual health (such as liver or kidney function) and exceeding them can cause serious harm.'
        : isPrescription
        ? 'I cannot translate, interpret, or confirm personal prescriptions or their dosages, as I am dedicated strictly to general health education.'
        : 'I cannot recommend or prescribe medications, suggest dosages, or advise on starting, stopping, or combining drugs.';

      const payload = makePayload(
        'medication_request',
        directAr,
        directEn,
        nextStepAr,
        nextStepEn,
        [
          'ما هو الأنسولين وكيف يعمل في الجسم بشكل عام؟',
          'ما المقصود بمقاومة المضادات الحيوية؟',
        ],
        [
          'What is insulin and how does it work in the body?',
          'What does antibiotic resistance mean in general?',
        ],
        true
      );
      if (classification.dynamicResponse?.trim()) {
        payload.answer.direct_answer = ensureNextStepLine(
          payload.answer.direct_answer,
          lang === 'ar' ? nextStepAr : nextStepEn,
          MENTIONS_CLINICIAN_NEXT_STEP_RE
        );
        payload.reply = payload.answer.direct_answer;
      }
      return { allowed: false, classification, earlyResponse: payload };
    }

    case 'OUT_OF_SCOPE': {
      const isBookingRequest = APPOINTMENT_BOOKING_RE.test(query);
      const bookingNextStepAr =
        'يمكنك الاستفادة من تبويب "دليل المريض" (Patient Guide) في صحتي للاستعداد لزيارة الطبيب، والتواصل مباشرة مع العيادة أو المستشفى لحجز الموعد.';
      const bookingNextStepEn =
        'You can use the Patient Guide tab in Sehati to help prepare for your doctor visit, and contact the clinic or hospital directly to book your appointment.';

      const dynamicOutAr = isBookingRequest
        ? `لا يمكنني حجز المواعيد الطبية أو إرسال تأكيدات للحجز، لأنني مساعد مخصص للتثقيف الصحي والمعلومات الطبية العامة فقط.\n\n${bookingNextStepAr}`
        : buildDynamicOutOfScopeFallback(query, 'ar');
      const dynamicOutEn = isBookingRequest
        ? `I cannot book clinical appointments or send confirmations, as I am dedicated strictly to general medical and health education.\n\n${bookingNextStepEn}`
        : buildDynamicOutOfScopeFallback(query, 'en');

      const payload = makePayload(
        'out_of_scope',
        dynamicOutAr,
        dynamicOutEn,
        '',
        '',
        [],
        [],
        false
      );

      if (isBookingRequest && classification.dynamicResponse?.trim()) {
        payload.answer.direct_answer = ensureNextStepLine(
          payload.answer.direct_answer,
          lang === 'ar' ? bookingNextStepAr : bookingNextStepEn,
          MENTIONS_PATIENT_GUIDE_RE
        );
        payload.reply = payload.answer.direct_answer;
      }

      return {
        allowed: false,
        classification,
        earlyResponse: payload,
      };
    }

    case 'AMBIGUOUS':
    default: {
      const clarifyingQ = buildClarifyingQuestionForQuery(query, lang);
      const snippet = (query || '').trim().slice(0, 60);
      const fallbackAr = snippet
        ? `لم يتضح لي الموضوع الطبي المقصود من "${snippet}". ${clarifyingQ}`
        : clarifyingQ;
      const fallbackEn = snippet
        ? `I'm not sure which health or medical topic you're referring to with "${snippet}". ${clarifyingQ}`
        : clarifyingQ;

      const payload = makePayload(
        'clarification_required',
        fallbackAr,
        fallbackEn,
        '',
        '',
        [],
        [],
        false
      );

      // Guarantee EVERY AMBIGUOUS response actually asks the user a clarifying question
      if (!/[؟?]/.test(payload.answer.direct_answer)) {
        payload.answer.direct_answer = `${payload.answer.direct_answer.trim()} ${clarifyingQ}`;
        payload.reply = payload.answer.direct_answer;
      }

      return {
        allowed: false,
        classification,
        earlyResponse: payload,
      };
    }
  }
}

/**
 * Synchronous deterministic safety filter (used by offline unit tests & legacy sync callers).
 */
export function runPreRetrievalSafetyFilter(
  query: string,
  history: ChatTurnHistory[],
  uiLang: Language,
  startTimeMs: number
): SafetyFilterDecision {
  const classification = classifyDomainHeuristic(query, history, uiLang);
  return buildSafetyEarlyDecision(query, classification, uiLang, startTimeMs);
}

/**
 * Mandatory Universal Semantic Safety Filter executed prior to response generation for EVERY user prompt.
 */
export async function runSemanticSafetyFilter(
  query: string,
  history: ChatTurnHistory[],
  uiLang: Language,
  startTimeMs: number,
  options: SemanticClassifierOptions = {}
): Promise<SafetyFilterDecision> {
  const classification = await classifyDomainSemantic(query, history, uiLang, options);
  return buildSafetyEarlyDecision(query, classification, uiLang, startTimeMs);
}
