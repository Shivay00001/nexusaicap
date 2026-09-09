export interface Provider {
  name: string;
  invoke(messages: any[], options?: any): Promise<any>;
}

export class BaseProvider implements Provider {
  name = 'base';
  async invoke(messages: any[], options?: any): Promise<any> {
    throw new Error('Not implemented');
  }
}
